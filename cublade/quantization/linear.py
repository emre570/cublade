import torch
import torch.nn as nn

from .quantizers import quantize_tensor
from .outlier_utils import separate_outliers
from cublade.bindings.triton import matmul as triton_matmul

class cuBladeW8A16LinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=torch.bfloat16,
        handle_outliers=False,
        outlier_threshold=6.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        
        # Standard quantized weights (always present)
        self.register_buffer("int8_weights", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=dtype))
        
        # Outlier handling buffers (only used if handle_outliers=True)
        self.register_buffer("outlier_weights", None)  # FP16 weights for outlier columns
        self.register_buffer("normal_cols", None)      # Indices of normal columns
        self.register_buffer("outlier_cols", None)     # Indices of outlier columns
        
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype)) if bias else None

    @torch.no_grad()
    def quantize(self, weight_fp: torch.Tensor):
        """
        Quantize the weight matrix.
        
        If handle_outliers=True, detects outlier columns and stores them separately
        in FP16, only quantizing the normal columns to INT8.
        """
        if self.handle_outliers:
            # LLM.int8 approach: detect and separate outlier columns
            weight_normal, weight_outlier, normal_cols, outlier_cols = separate_outliers(
                weight_fp, threshold=self.outlier_threshold
            )
            
            # Store outlier information
            self.outlier_weights = weight_outlier  # Keep outliers in original dtype
            self.normal_cols = torch.tensor(normal_cols, dtype=torch.long)
            self.outlier_cols = torch.tensor(outlier_cols, dtype=torch.long)
            
            # Quantize only the normal columns
            qt = quantize_tensor(
                weight_normal, scheme="symmetric", dtype=torch.int8, mode="channel", ch_axis=0
            )
            
            # Resize int8_weights buffer to match normal columns only
            self.int8_weights = qt.data
        else:
            # Standard quantization: quantize entire weight matrix
            qt = quantize_tensor(
                weight_fp, scheme="symmetric", dtype=torch.int8, mode="channel", ch_axis=0
            )
            self.int8_weights.copy_(qt.data)
        
        # qt is cuBlade's QuantizedTensor wrapper
        assert torch.allclose(qt.zero_point, torch.zeros_like(qt.zero_point)), "Expect symmetric zp=0"
        
        # Store scales (per-channel along out_features)
        scale_1d = qt.scale.squeeze()
        assert scale_1d.ndim == 1 and scale_1d.numel() == self.out_features
        self.scales.copy_(scale_1d.to(self.scales.dtype))

    def forward(self, x: torch.Tensor):
        """
        Forward pass with optional outlier handling.
        
        If handle_outliers=True, performs dual matmul:
        - Normal features × INT8 weights (quantized)
        - Outlier features × FP16 weights (preserved)
        
        Otherwise, standard quantized matmul.
        """
        orig_shape = x.shape
        x2d = x.view(-1, self.in_features).contiguous()
        dev = x2d.device

        if self.handle_outliers and self.outlier_cols is not None and len(self.outlier_cols) > 0:
            # LLM.int8 mixed-precision path
            
            # Split input features into normal and outlier columns
            x_normal = x2d[:, self.normal_cols]    # [batch, in_features - num_outliers]
            x_outlier = x2d[:, self.outlier_cols]  # [batch, num_outliers]
            
            # Quantized path: normal features × INT8 weights
            W_normal = self.int8_weights.to(dtype=x2d.dtype, device=dev).contiguous()
            
            if dev.type == "cuda":
                y_normal = triton_matmul(x_normal, W_normal.t())
            else:
                y_normal = x_normal @ W_normal.t()
            
            # Apply scales to quantized output
            y_normal = y_normal * self.scales.to(dtype=x2d.dtype, device=dev)
            
            # FP16 path: outlier features × FP16 weights
            W_outlier = self.outlier_weights.to(dtype=x2d.dtype, device=dev)
            y_outlier = x_outlier @ W_outlier.t()
            
            # Combine both paths
            y2d = y_normal + y_outlier
        else:
            # Standard path: no outlier handling
            W = self.int8_weights.to(dtype=x2d.dtype, device=dev).contiguous()

            if dev.type == "cuda":
                y2d = triton_matmul(x2d, W.t())
            else:
                y2d = x2d @ W.t()

            # Apply per-channel scales
            y2d = y2d * self.scales.to(dtype=y2d.dtype, device=dev)

        # Add bias
        if self.bias is not None:
            y2d = y2d + self.bias.to(y2d.dtype)

        return y2d.view(*orig_shape[:-1], self.out_features)