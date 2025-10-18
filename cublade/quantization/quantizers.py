import torch
import torch.nn as nn
from typing import Optional, List

from .utils import _qrange, compute_params, replace_linear_layers
from .tensor import QuantizedTensor

# ---------------------------
# Quantize
# ---------------------------
def quantize_int8(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype = torch.int8,
):
    """Affine quantize a tensor into int8/uint8 storage.

    Parameters
    ----------
    x : torch.Tensor
        Source tensor expressed in floating point.
    scale : torch.Tensor
        Scaling factors broadcastable to ``x``.
    zero_point : torch.Tensor
        Zero-point offsets broadcastable to ``x``.
    dtype : torch.dtype, default torch.int8
        Destination integer dtype (``torch.int8`` or ``torch.uint8``).

    Returns
    -------
    torch.Tensor
        Integer tensor with the same shape as ``x`` and values clamped to the
        representable range of ``dtype``.
    """
    q_min, q_max = _qrange(dtype)
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, min=float(q_min), max=float(q_max))
    return q.to(dtype)

def quantize_per_group_int8(
    x: torch.Tensor,
    group_size: int,
    axis: int = -1,
    dtype: torch.dtype = torch.int8,
    eps: float = 1e-12,
):
    """Symmetric per-group INT8/UINT8 quantization.

    The target ``axis`` is grouped into contiguous blocks of ``group_size``
    elements. Each group is quantized independently using symmetric affine
    parameters (zero-point becomes a tensor of zeros for int8, or 128 for
    uint8).

    Parameters
    ----------
    x : torch.Tensor
        Source tensor.
    group_size : int
        Number of consecutive elements per quantization group; must evenly
        divide the length of ``axis``.
    axis : int, default -1
        Axis to group and quantize.
    dtype : torch.dtype, default torch.int8
        Destination integer dtype (``torch.int8`` or ``torch.uint8``).
    eps : float, default 1e-12
        Numerical lower bound to avoid zero division.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q, scale, zero_point)`` where ``q`` has the same shape as ``x`` and
        ``scale``/``zero_point`` contain one entry per quantization group.
    """
    # Move target axis to last for contiguous grouping
    x_perm = x.movedim(axis, -1).contiguous()
    *lead, C = x_perm.shape
    assert C % group_size == 0, "group_size must evenly divide the target axis."

    xg = x_perm.view(-1, group_size)
    scale, zp = compute_params(xg, scheme="symmetric", dtype=dtype, per_channel=True, ch_axis=0, eps=eps)
    qg = quantize_int8(xg, scale, zp, dtype=dtype)

    # Restore original shape/axis
    q_perm = qg.view(*lead, C)
    q = q_perm.movedim(-1, axis)

    # 'scale' is per-group flat vector aligned with rows of xg; keep as 1D for simplicity
    return q, scale, zp

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

    Parameters
    ----------
    x : torch.Tensor
        Floating-point tensor to be quantized.
    scheme : str, default "symmetric"
        Quantization scheme, currently ``"symmetric"`` or ``"asymmetric"`` for
        per-tensor/per-channel modes. Per-group mode always uses symmetric
        quantization.
    dtype : torch.dtype, default torch.int8
        Integer storage type (``torch.int8`` or ``torch.uint8``).
    mode : str, default "tensor"
        Quantization granularity: ``"tensor"`` (per-tensor), ``"channel"``
        (per-channel along ``ch_axis``), or ``"group"`` (per-group along
        ``ch_axis`` using ``group_size``).
    group_size : int, default 32
        Group size for ``mode="group"``; must evenly divide the length of the
        grouped axis.
    ch_axis : int, default 0
        Channel axis for per-channel or per-group quantization.
    eps : float, default 1e-12
        Numerical lower bound used when computing scales.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Quantized tensor and its ``(scale, zero_point)`` parameters.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``{"tensor", "channel", "group"}``.
    """
    assert dtype in (torch.int8, torch.uint8), "Only int8/uint8 are supported."
    
    #For per-tensor
    if mode == 'tensor':
        if dtype in (torch.int8, torch.uint8):
            scale, zp = compute_params(
                x, scheme=scheme, dtype=dtype, per_channel=False, ch_axis=ch_axis, eps=eps
            )
            q = quantize_int8(x, scale, zp, dtype=dtype)
    elif mode == 'channel':
        if dtype in (torch.int8, torch.uint8):
            scale, zp = compute_params(
                x, scheme=scheme, dtype=dtype, per_channel=True, ch_axis=ch_axis, eps=eps
            )
            q = quantize_int8(x, scale, zp, dtype=dtype)
    elif mode == 'group':
        if dtype in (torch.int8, torch.uint8):
            q, scale, zp = quantize_per_group_int8(
                x, group_size=group_size, dtype=dtype, axis=ch_axis, eps=eps)
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")
    
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