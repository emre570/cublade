import torch

from .tensor import QuantizedTensor
from cublade.bindings.cuda.dequantize_fp8 import (
    dequantize_per_tensor_fp8 as _cuda_dequant_fp8_tensor,
)
from cublade.bindings.cuda.dequantize_int8 import (
    dequantize_per_tensor_int8 as _cuda_dequant_int8_tensor,
    dequantize_per_channel_int8 as _cuda_dequant_int8_channel,
    dequantize_per_group_int8 as _cuda_dequant_int8_group,
)


_INT8_DTYPES = (torch.int8,)
_FP8_DTYPES = (torch.float8_e4m3fn,)


def _require_cuda_dequant_input(
    q: torch.Tensor, out_dtype: torch.dtype, *, what: str
) -> None:
    if not q.is_cuda:
        raise ValueError(f"{what}: tensor must be on CUDA, got device={q.device}")
    if not q.is_contiguous():
        raise ValueError(f"{what}: tensor must be contiguous")
    if q.dtype not in _INT8_DTYPES + _FP8_DTYPES:
        raise ValueError(
            f"{what}: q.dtype must be torch.int8 or torch.float8_e4m3fn, got {q.dtype}"
        )
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"{what}: out_dtype must be float16 or bfloat16, got {out_dtype}"
        )


# ---------------------------
# High-level wrapper
# ---------------------------
def dequantize_tensor(qt: QuantizedTensor):
    """High-level entry point that consumes ``QuantizedTensor`` metadata."""
    _require_cuda_dequant_input(qt.data, qt.original_dtype, what="dequantize_tensor")

    if qt.dtype == torch.float8_e4m3fn:
        if qt.mode != "tensor":
            raise NotImplementedError(
                "FP8 dequant only supports mode='tensor'"
            )
        return _cuda_dequant_fp8_tensor(qt.data, qt.scale, qt.original_dtype)

    if qt.dtype == torch.int8:
        if qt.mode == "tensor":
            return _cuda_dequant_int8_tensor(qt.data, qt.scale, qt.original_dtype)
        if qt.mode == "channel":
            if qt.axis != 0:
                raise NotImplementedError(
                    "INT8 channel dequant requires axis=0; got axis=" + str(qt.axis)
                )
            return _cuda_dequant_int8_channel(qt.data, qt.scale, qt.original_dtype)
        if qt.mode == "group":
            if qt.group_size is None:
                raise ValueError("INT8 group dequant needs qt.group_size")
            q_perm = qt.data.movedim(qt.axis, -1).contiguous()
            lead = q_perm.shape[:-1]
            F = q_perm.shape[-1]
            if F % qt.group_size != 0:
                raise ValueError(
                    f"group_size {qt.group_size} must evenly divide axis length {F}"
                )
            q_grouped = q_perm.view(-1, qt.group_size).contiguous()
            y_grouped = _cuda_dequant_int8_group(
                q_grouped, qt.scale, qt.group_size, qt.original_dtype
            )
            return y_grouped.view(*lead, F).movedim(-1, qt.axis).contiguous()
        raise ValueError(f"Unsupported INT8 mode '{qt.mode}'")

    raise ValueError(f"No dequant kernel registered for dtype={qt.dtype}")
