from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fouroversix import DataType, ModelQuantizationConfig, QuantizedTensor, ScaleRule
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model


def _load_fouroversix_model_config(config: Qwen3Config) -> ModelQuantizationConfig:
    quant_config = dict(
        getattr(config, "fouroversix_awq_weight_only_config", None)
        or getattr(config, "quantization_config", {})
        or {}
    )
    return ModelQuantizationConfig(**quant_config)


class SGLFourOverSixWeightOnlyLinear(nn.Module):
    """Weight-only FP4 linear layer that stays invisible to SGLang's nn.Linear rewrite."""

    def __init__(
        self,
        module: nn.Linear,
        fq_config,
    ) -> None:
        super().__init__()
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.config = fq_config

        if module.bias is not None:
            self.bias = nn.Parameter(
                torch.empty_like(module.bias, device=module.bias.device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "quantized_weight_values",
            torch.zeros(
                self.out_features,
                self.in_features // 2,
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "quantized_weight_scale_factors",
            torch.zeros(
                self.out_features
                * self.in_features
                // self.config.dtype.block_size(),
                dtype=self.config.dtype.scale_dtype(),
            ),
        )
        self.register_buffer(
            "quantized_weight_amax",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "quantized_weight_metadata",
            torch.zeros(4, dtype=torch.int32),
        )

    def _quantized_weight_tensor(self) -> QuantizedTensor:
        cached = getattr(self, "_quantized_weight", None)
        if cached is None:
            metadata = self.quantized_weight_metadata.tolist()
            original_shape = tuple(int(v) for v in metadata[:2])
            padded_shape = tuple(int(v) for v in metadata[2:])
            cached = QuantizedTensor(
                self.quantized_weight_values,
                self.quantized_weight_scale_factors,
                self.quantized_weight_amax,
                self.config.dtype,
                original_shape,
                self.config.weight_scale_rule,
                padded_shape,
            )
            self._quantized_weight = cached
        return cached

    def _dequantized_weight_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cached = getattr(self, "_dequantized_weight", None)
        if cached is None or cached.dtype != dtype or cached.device != device:
            cached = self._quantized_weight_tensor().dequantize(dtype=dtype).to(device=device)
            self._dequantized_weight = cached
        return cached

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._dequantized_weight_tensor(dtype=input.dtype, device=input.device)
        out = F.linear(input, weight, self.bias)
        output_dtype = self.config.output_dtype.torch_dtype()
        return out if out.dtype == output_dtype else out.to(output_dtype)


def _replace_linear_modules(
    module: nn.Module,
    fq_model_config: ModelQuantizationConfig,
    prefix: str,
) -> None:
    for child_name, child_module in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if (
            isinstance(child_module, nn.Linear)
            and full_name not in fq_model_config.modules_to_not_convert
        ):
            setattr(
                module,
                child_name,
                SGLFourOverSixWeightOnlyLinear(
                    child_module,
                    fq_model_config.get_module_config(full_name),
                ),
            )
            continue

        _replace_linear_modules(child_module, fq_model_config, full_name)


class Qwen3ModelFourOverSixAWQWeightOnly(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        fq_model_config = _load_fouroversix_model_config(config)
        _replace_linear_modules(self, fq_model_config, "model")


class Qwen3ForCausalLMFourOverSixAWQWeightOnly(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        fq_model_config = _load_fouroversix_model_config(config)
        _replace_linear_modules(self.model, fq_model_config, "model")


def patch_config_json(model_dir: str | Path) -> None:
    """Patch config.json so HF/SGLang can load the custom remote-code model."""

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Qwen3ForCausalLMFourOverSixAWQWeightOnly"]
    auto_map = dict(config.get("auto_map", {}) or {})
    auto_map["AutoModel"] = (
        "modeling_qwen3_fouroversix_awq_weight_only.Qwen3ModelFourOverSixAWQWeightOnly"
    )
    auto_map["AutoModelForCausalLM"] = (
        "modeling_qwen3_fouroversix_awq_weight_only.Qwen3ForCausalLMFourOverSixAWQWeightOnly"
    )
    config["auto_map"] = auto_map
    config_path.write_text(json.dumps(config, indent=2) + "\n")
