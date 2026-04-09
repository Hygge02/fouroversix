from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import ModelQuantizationConfig


class QuantizedModule:
    """Base class for all quantized modules."""

    _registry: ClassVar[dict[type[nn.Module], type[nn.Module]]] = {}
    _should_replace_existing_modules_in_model: ClassVar[dict[type[nn.Module], bool]] = (
        {}
    )

    @classmethod
    def is_quantized_module_type(cls, module_type: type[nn.Module]) -> bool:
        """Return True if the given module type is a quantized module."""
        return module_type in cls._registry.values()

    @classmethod
    def get_cls(
        cls,
        high_precision_cls: type[nn.Module],
    ) -> type[nn.Module] | None:
        """Get the quantized module for a given high-precision module."""
        return cls._registry.get(high_precision_cls)

    @classmethod
    def should_replace_existing_modules_in_model(
        cls,
        module_type: type[nn.Module],
    ) -> bool:
        """Determine whether module should be replaced."""
        return cls._should_replace_existing_modules_in_model.get(module_type, False)

    @classmethod
    def register(
        cls,
        high_precision_cls: type[nn.Module],
        *,
        replace_existing_modules_in_registry: bool = False,
        replace_existing_modules_in_model: bool = True,
    ) -> Callable[[type[nn.Module]], type[nn.Module]]:
        """
        Register a new type of quantized module.

        Args:
            high_precision_cls: (`type[nn.Module]`): The high precision module to be
            mapped to a fouroversix quantized module.
            replace_existing_modules_in_registry (bool): determines whether we should
            replace the existing module in the registry.
            replace_existing_modules_in_model (bool): determines whether we should
            replace the existing module in the model including the weights.

        """

        if (
            high_precision_cls in cls._registry
            and not replace_existing_modules_in_registry
        ):
            msg = f"High-precision module {high_precision_cls} is already registered."
            raise ValueError(msg)

        modules_to_delete = []

        for module_cls in cls._registry:
            if high_precision_cls is not None and issubclass(
                high_precision_cls,
                module_cls,
            ):
                if replace_existing_modules_in_registry:
                    modules_to_delete.append(module_cls)
                else:
                    msg = (
                        f"High-precision module {high_precision_cls} is a subclass of "
                        f"{module_cls}, which is already registered."
                    )
                    raise TypeError(msg)

        for module_cls in modules_to_delete:
            del cls._registry[module_cls]

        def inner_wrapper(
            wrapped_cls: type[nn.Module],
        ) -> type[nn.Module]:
            cls._registry[high_precision_cls] = wrapped_cls
            cls._should_replace_existing_modules_in_model[high_precision_cls] = (
                replace_existing_modules_in_model
            )
            return wrapped_cls

        return inner_wrapper


def quantize_model(
    model: nn.Module,
    config: ModelQuantizationConfig,
    **kwargs: dict[str, Any],
) -> None:
    for module_name, module in model.named_modules():
        if (
            module_name == ""
            or module_name in config.modules_to_not_convert
            or not isinstance(module, nn.Module)
        ):
            continue

        module_cls = QuantizedModule.get_cls(type(module))
        should_replace = QuantizedModule.should_replace_existing_modules_in_model(
            type(module),
        )

        if module_cls is None or not should_replace:
            continue

        quantized_module = module_cls(
            module,
            config.get_module_config(module_name),
            **kwargs,
        )
        model.set_submodule(module_name, quantized_module)


def apply_offline_weight_quantization(model: nn.Module) -> None:
    """
    Quantize registered module weights offline and remove the master weights.

    Modules participate in this pass by exposing `parameters_to_quantize` and
    `get_quantized_parameters`.
    """

    for module in model.modules():
        parameter_names = getattr(module, "parameters_to_quantize", ())

        if not parameter_names:
            continue

        for parameter_name in parameter_names:
            if not hasattr(module, parameter_name):
                continue

            parameter = getattr(module, parameter_name)
            if parameter is None:
                continue

            quantized_parameters = module.get_quantized_parameters(
                parameter_name,
                parameter.data,
            )

            for quantized_name, quantized_value in quantized_parameters.items():
                if hasattr(module, quantized_name):
                    existing = getattr(module, quantized_name)
                    if torch.is_tensor(existing):
                        existing.data = quantized_value.to(
                            device=existing.device,
                            dtype=existing.dtype,
                        )
                    else:
                        setattr(module, quantized_name, quantized_value)
                else:
                    module.register_buffer(quantized_name, quantized_value)

            delattr(module, parameter_name)

        if hasattr(module, "config"):
            module.config.keep_master_weights = False

        for cache_name in (
            "_dequantized_weight",
            "_quantized_weight",
            "_quantized_weights",
        ):
            if hasattr(module, cache_name):
                delattr(module, cache_name)
