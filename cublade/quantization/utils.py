import torch.nn as nn
from typing import Optional, List, Type, Dict, Any


# ---------------------------
# Replace linear layers
# ---------------------------
def replace_linear_layers(
    module: nn.Module,
    target_class: Type[nn.Module],
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    quant_kwargs: Optional[Dict[str, Any]] = None,
    _parent_name: str = "",
):
    """
    Recursively replace nn.Linear with target_class.

    Internal utility used by quantize_model().
    """
    quant_kwargs = quant_kwargs or {}
    exclude_modules = exclude_modules or []

    for name, child in list(module.named_children()):
        full_name = f"{_parent_name}.{name}" if _parent_name else name

        # Check if this module should be quantized
        if isinstance(child, nn.Linear):
            # Skip if in exclude list
            if any(excl in full_name for excl in exclude_modules):
                continue

            # Skip if target_modules specified and this isn't in it
            if target_modules is not None:
                if not any(tgt in full_name for tgt in target_modules):
                    continue

            # Replace with quantized layer
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(
                child.in_features,
                child.out_features,
                bias=old_bias is not None,
                dtype=child.weight.dtype,
                **quant_kwargs,  # Pass quant-specific options
            )

            # Set module first (important for device placement)
            setattr(module, name, new_module)

            # Quantize the weights after setattr
            getattr(module, name).quantize(old_weight)

            # Assign bias if it exists (after setattr)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recurse into child modules
            replace_linear_layers(
                child,
                target_class,
                target_modules,
                exclude_modules,
                quant_kwargs,
                _parent_name=full_name,
            )
