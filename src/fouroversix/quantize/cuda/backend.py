import random

import torch
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.utils import (
    BLACKWELL_SM_IDS,
    DataType,
    QuantizeBackend,
    RoundStyle,
    get_effective_major_compute_capability,
)


class CUDAQuantizeBackend(QuantizeBackendBase):
    """
    The CUDA quantization backend. Supports basic quantization options (no 2D block
    scaling, no stochastic rounding, no random Hadamard transform). As a result, it can
    be used for inference, but not training. Requires a Blackwell GPU.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the CUDA backend is available on the current machine."""

        if (
            not torch.cuda.is_available()
            or get_effective_major_compute_capability() not in BLACKWELL_SM_IDS
        ):
            return False

        try:
            import fouroversix._C  # noqa: F401
        except ModuleNotFoundError:
            return False

        return True

    @classmethod
    def can_quantize(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the CUDA backend supports the given input and quantization
        configuration.
        """

        if not super().can_quantize(x, config):
            return False

        return (
            x.device.type == "cuda"
            and x.dtype in {torch.float16, torch.bfloat16}
            and config.round_style == RoundStyle.nearest
            and config.dtype == DataType.nvfp4
            and not config.transpose
            and not config.pseudo_quantize
        )

    @classmethod
    def _get_rbits(cls, config: QuantizationConfig) -> int:
        rbits = config.kwargs.get("rbits")
        if rbits is not None:
            return int(rbits)

        if config.round_style == RoundStyle.stochastic:
            return random.getrandbits(31)

        return -1

    @classmethod
    def quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the CUDA backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        from .ops import quantize

        values, scale_factors, amax = quantize(
            x,
            config.dtype == DataType.nvfp4,
            config.round_style == RoundStyle.nearest,
            config.rht,
            config.block_scale_2d,
            config.transpose,
            config.scale_rule.cuda_id,
            cls._get_rbits(config),
        )

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
            config.round_style,
        )


class CUDAFusedQuantizeBackend(CUDAQuantizeBackend):
    """
    TE-style fused CUDA backend for FourOverSix quantization.

    The existing C++/CUDA kernel supports the heavy FourOverSix path: NVFP4,
    MSE/MAE/abs-max dynamic 4/6 selection, optional RHT, optional 2D scales, nearest
    rounding, and SM100 stochastic rounding. For transposed non-2D paths, the CUDA
    kernel reads the transposed view directly while producing the same quantized layout;
    this avoids a separate high-precision `x.t().contiguous()` workspace.
    """

    @classmethod
    def _can_use_fused_cuda(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if not QuantizeBackendBase.can_quantize.__func__(cls, x, config):
            return False

        if x.device.type != "cuda" or x.dtype not in {torch.float16, torch.bfloat16}:
            return False

        if config.dtype != DataType.nvfp4 or config.pseudo_quantize:
            return False

        effective_n = x.shape[0] if config.transpose else x.shape[1]
        if effective_n % 64 != 0:  # noqa: PLR2004
            return False

        if config.transpose and config.block_scale_2d:
            return False

        supported_round_styles = {RoundStyle.nearest, RoundStyle.stochastic}
        if config.round_style not in supported_round_styles:
            return False

        if (
            config.round_style == RoundStyle.stochastic
            and get_effective_major_compute_capability() != 10  # noqa: PLR2004
        ):
            return False

        return True

    @classmethod
    def can_quantize(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if cls._can_use_fused_cuda(x, config):
            return True

        from fouroversix.quantize.triton import TritonQuantizeBackend

        return TritonQuantizeBackend.can_quantize(x, config)

    @classmethod
    def quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        if cls._can_use_fused_cuda(x, config):
            return super().quantize(x, config)

        fallback_config = QuantizationConfig(
            backend=QuantizeBackend.triton,
            block_scale_2d=config.block_scale_2d,
            dtype=config.dtype,
            kwargs=dict(config.kwargs),
            pseudo_quantize=config.pseudo_quantize,
            rht=config.rht,
            round_style=config.round_style,
            scale_rule=config.scale_rule,
            transpose=config.transpose,
        )

        from fouroversix.quantize.triton import TritonQuantizeBackend

        return TritonQuantizeBackend.quantize(x, fallback_config)
