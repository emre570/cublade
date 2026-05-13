import torch
import torch.nn as nn
from typing import Optional, List

from .utils import replace_linear_layers
from .tensor import QuantizedTensor
from cublade.bindings.cuda.quantize_fp8 import (
    quantize_per_tensor_fp8 as _cuda_quant_fp8_tensor,
)
from cublade.bindings.cuda.quantize_int8 import (
    quantize_per_tensor_int8 as _cuda_quant_int8_tensor,
    quantize_per_channel_int8 as _cuda_quant_int8_channel,
    quantize_per_group_int8 as _cuda_quant_int8_group,
)

# ---------------------------
# CUDA kernel registry (mode, dtype) -> kernel callable.
# Per-tensor kernels accept (x, eps); per-channel accepts (x, eps); per-group
# accepts (x_grouped, group_size, eps). The dispatcher below threads the right
# args per mode. Each kernel returns (q, scale).
# ---------------------------
_CUDA_QUANT_KERNELS = {
    ("tensor", torch.float8_e4m3fn): _cuda_quant_fp8_tensor,
    ("tensor", torch.int8):          _cuda_quant_int8_tensor,
    ("channel", torch.int8):         _cuda_quant_int8_channel,
    ("group", torch.int8):           _cuda_quant_int8_group,
}

# Per-(mode, dtype) allowed input dtypes. FP8 stays strict (fp16/bf16);
# INT8 also accepts fp32 because nn.Linear.weight is fp32 by default and
# casting away precision before quant would silently degrade the scales.
_ALLOWED_INPUT_DTYPES = {
    ("tensor", torch.float8_e4m3fn): {torch.float16, torch.bfloat16},
    ("tensor", torch.int8):          {torch.float16, torch.bfloat16, torch.float32},
    ("channel", torch.int8):         {torch.float16, torch.bfloat16, torch.float32},
    ("group", torch.int8):           {torch.float16, torch.bfloat16, torch.float32},
}


def _require_cuda_input(x: torch.Tensor, key, *, what: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{what}: tensor must be on CUDA, got device={x.device}")
    if not x.is_contiguous():
        raise ValueError(f"{what}: tensor must be contiguous")
    allowed = _ALLOWED_INPUT_DTYPES[key]
    if x.dtype not in allowed:
        raise ValueError(
            f"{what}: tensor dtype must be one of {sorted(str(d) for d in allowed)}, "
            f"got {x.dtype}"
        )


def _zero_scalar(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


# ---------------------------
# High-level wrapper
# ---------------------------
def quantize_tensor(
    x: torch.Tensor,
    scheme: str = "symmetric",
    dtype: torch.dtype = torch.int8,
    mode: str = 'tensor',
    group_size: int = 32,
    ch_axis: int = 0,
    eps: float = 1e-12,
):
    """Quantize ``x`` according to the requested granularity.

    All paths dispatch to CUDA kernels. ``x`` must be a contiguous CUDA
    tensor; INT8 paths accept fp16/bf16/fp32 inputs, FP8 accepts fp16/bf16.
    Symmetric scheme only.

    Parameters
    ----------
    x : torch.Tensor
        Floating-point tensor to be quantized.
    scheme : str, default "symmetric"
        Only ``"symmetric"`` is supported.
    dtype : torch.dtype, default torch.int8
        Quantized storage dtype: ``torch.int8`` or ``torch.float8_e4m3fn``.
    mode : str, default "tensor"
        Granularity: ``"tensor"`` (single scale), ``"channel"`` (per-row
        scale along ``ch_axis``), or ``"group"`` (per-group scale along
        ``ch_axis`` with ``group_size``). FP8 supports ``"tensor"`` only.
    group_size : int, default 32
        Group size for ``mode="group"``. Must divide the grouped axis length.
    ch_axis : int, default 0
        Channel axis for ``mode="channel"`` (INT8 enforces 0) or ``"group"``.
    eps : float, default 1e-12
        Numerical floor on amax.

    Returns
    -------
    QuantizedTensor
        Wrapper bundling ``(data, scale, zero_point, mode, axis, group_size,
        dtype, original_dtype)``. ``zero_point`` is a 0-d zero scalar for the
        symmetric path.
    """
    if scheme != "symmetric":
        raise NotImplementedError(
            f"quantize_tensor: only scheme='symmetric' is supported, got '{scheme}'"
        )

    key = (mode, dtype)
    kernel = _CUDA_QUANT_KERNELS.get(key)
    if kernel is None:
        raise ValueError(
            f"No quant kernel registered for mode='{mode}', dtype={dtype}"
        )
    _require_cuda_input(x, key, what="quantize_tensor")

    if mode == "tensor":
        q, scale = kernel(x, eps)
    elif mode == "channel":
        if ch_axis != 0:
            raise NotImplementedError(
                "INT8 channel quant requires ch_axis=0; got ch_axis=" + str(ch_axis)
            )
        q, scale = kernel(x, eps)
    elif mode == "group":
        x_perm = x.movedim(ch_axis, -1).contiguous()
        lead = x_perm.shape[:-1]
        F = x_perm.shape[-1]
        if F % group_size != 0:
            raise ValueError(
                f"group_size {group_size} must evenly divide axis length {F}"
            )
        x_grouped = x_perm.view(-1, group_size).contiguous()
        q_grouped, scale = kernel(x_grouped, group_size, eps)
        q = q_grouped.view(*lead, F).movedim(-1, ch_axis).contiguous()
    else:
        raise ValueError(
            f"Unsupported mode '{mode}'; expected tensor / channel / group"
        )

    zp = _zero_scalar(x.device)
    return QuantizedTensor(q, scale, zp, mode, ch_axis, group_size, dtype, x.dtype)


QUANT_LAYER_REGISTRY = {}


def _init_registry():
    """Lazy initialization to avoid circular imports."""
    if not QUANT_LAYER_REGISTRY:
        from .linear import cuBladeW8A16LinearLayer
        QUANT_LAYER_REGISTRY["w8a16"] = cuBladeW8A16LinearLayer


def quantize_model(
    model: nn.Module,
    quant_type: str = "w8a16",
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    inplace: bool = True,
    **quant_kwargs,
):
    """
    Quantize a model's linear layers to reduce memory footprint.

    Replaces `nn.Linear` layers with quantized equivalents based on
    the specified quantization scheme.

    Args:
        model: PyTorch model to quantize
        quant_type: Quantization scheme - one of:
            - "w8a16": INT8 weights, FP16 activations (default)
            - "w8a8": INT8 weights, INT8 activations (future)
            - "w4a16": INT4 weights, FP16 activations (future)
        target_modules: List of module names to quantize. If None, quantize
            all Linear layers except those in `exclude_modules`.
            Example: ["q_proj", "k_proj", "v_proj"] for attention only.
        exclude_modules: List of module names to skip.
            Example: ["lm_head", "embed_tokens"]
        inplace: If True, modify model in place. Otherwise, return a copy.
        **quant_kwargs: Additional options passed to the quantized layer:
            - For W8A16:
                - handle_outliers (bool): Enable LLM.int8 outlier handling
                - outlier_threshold (float): Threshold for outlier detection
                - dtype (torch.dtype): Activation dtype

    Returns:
        Quantized model (same object if inplace=True)

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from cublade.quantization import quantize_model
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model = quantize_model(
        ...     model,
        ...     quant_type="w8a16",
        ...     exclude_modules=["lm_head"]
        ... )
        >>> # Model is now quantized, use normally
        >>> output = model(input_ids)
    """
    _init_registry()

    if quant_type not in QUANT_LAYER_REGISTRY:
        available = ", ".join(QUANT_LAYER_REGISTRY.keys())
        raise ValueError(
            f"Unknown quant_type '{quant_type}'. Available: {available}"
        )

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)

    # Get the appropriate quantized layer class
    quant_layer_cls = QUANT_LAYER_REGISTRY[quant_type]

    # Perform replacement
    replace_linear_layers(
        model,
        target_class=quant_layer_cls,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        quant_kwargs=quant_kwargs,
    )

    return model
