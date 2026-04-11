#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/root/fouroversix/artifacts/qwen3-8b-awq-fouroversix-weight-only",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/fouroversix/artifacts/qwen3-8b-awq-fouroversix-offline-dequantized-bf16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def is_weight_only_linear(module: nn.Module) -> bool:
    return module.__class__.__name__ == "SGLFourOverSixWeightOnlyLinear"


def replace_weight_only_modules(module: nn.Module) -> tuple[int, int]:
    replaced = 0
    skipped = 0
    for child_name, child_module in list(module.named_children()):
        if is_weight_only_linear(child_module):
            device = child_module.quantized_weight_values.device
            dtype = torch.bfloat16
            new_module = nn.Linear(
                child_module.in_features,
                child_module.out_features,
                bias=child_module.bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                weight = child_module._quantized_weight_tensor().dequantize(dtype=dtype)
                new_module.weight.copy_(weight)
                if child_module.bias is not None:
                    new_module.bias.copy_(child_module.bias.to(dtype=dtype))
            setattr(module, child_name, new_module)
            replaced += 1
            del child_module
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        child_replaced, child_skipped = replace_weight_only_modules(child_module)
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped


def patch_config_json(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Qwen3ForCausalLM"]
    config.pop("auto_map", None)
    config.pop("fouroversix_awq_weight_only_config", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=False,
    )
    model.eval()

    replaced, skipped = replace_weight_only_modules(model)
    print(json.dumps({"event": "replaced_modules", "count": replaced, "skipped": skipped}))

    model.config.architectures = ["Qwen3ForCausalLM"]
    if hasattr(model.config, "auto_map"):
        model.config.auto_map = None
    if hasattr(model.config, "fouroversix_awq_weight_only_config"):
        model.config.fouroversix_awq_weight_only_config = None

    tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)

    model = model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    patch_config_json(output_dir)

    print(json.dumps({"event": "saved", "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
