import torch
import torch.nn.functional as F
from fouroversix import (
    DataType,
    ModuleQuantizationConfig,
    QuantizeBackend,
    ScaleRule,
    quantize_to_fp4,
)
from fouroversix.model import apply_offline_weight_quantization
from fouroversix.model.modules import FourOverSixWeightOnlyLinear


def test_offline_weight_only_linear_matches_dequantized_weight_reference() -> None:
    torch.manual_seed(0)

    linear = torch.nn.Linear(64, 128, bias=True, dtype=torch.bfloat16)
    config = ModuleQuantizationConfig(
        activation_scale_rule=ScaleRule.static_4,
        dtype=DataType.nvfp4,
        quantize_backend=QuantizeBackend.pytorch,
        weight_scale_rule=ScaleRule.mse,
    )

    weight_only_linear = FourOverSixWeightOnlyLinear(linear, config)
    apply_offline_weight_quantization(weight_only_linear)

    x = torch.randn(8, 64, dtype=torch.bfloat16) * 32

    expected_weight = quantize_to_fp4(
        linear.weight.detach(),
        config.get_weight_config(),
    ).dequantize(dtype=x.dtype)
    expected = F.linear(x, expected_weight, linear.bias)

    assert not hasattr(weight_only_linear, "weight")
    torch.testing.assert_close(weight_only_linear(x), expected, atol=0, rtol=0)


def test_offline_weight_only_linear_can_be_materialized_from_master_weights() -> None:
    torch.manual_seed(0)

    linear = torch.nn.Linear(64, 128, bias=True, dtype=torch.bfloat16)
    config = ModuleQuantizationConfig(
        dtype=DataType.nvfp4,
        keep_master_weights=True,
        quantize_backend=QuantizeBackend.pytorch,
        weight_scale_rule=ScaleRule.mse,
    )

    weight_only_linear = FourOverSixWeightOnlyLinear(linear, config)
    apply_offline_weight_quantization(weight_only_linear)

    x = torch.randn(8, 64, dtype=torch.bfloat16)
    expected_weight = quantize_to_fp4(
        linear.weight.detach(),
        config.get_weight_config(),
    ).dequantize(dtype=x.dtype)

    assert not hasattr(weight_only_linear, "weight")
    torch.testing.assert_close(
        weight_only_linear(x),
        F.linear(x, expected_weight, linear.bias),
        atol=0,
        rtol=0,
    )
