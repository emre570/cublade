"""
Outlier detection and handling utilities for LLM.int8-style quantization.

Based on the LLM.int8 paper approach: detect systematic outliers in specific
input features (columns) and handle them separately in mixed precision.
"""

import torch
from typing import Tuple, List


def detect_outlier_columns(weight: torch.Tensor, threshold: float = 6.0) -> List[int]:
    """
    Detect outlier columns (input features) in weight matrix using LLM.int8 approach.
    
    A column is marked as an outlier if ANY value in that column exceeds the threshold.
    This matches the paper's observation that certain input features systematically
    produce large activations.
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        threshold: Absolute value threshold for outlier detection (default: 6.0)
    
    Returns:
        List of column indices that are outliers (typically 0.1-1% of columns)
    
    Example:
        >>> weight = torch.randn(768, 768)
        >>> outlier_cols = detect_outlier_columns(weight, threshold=6.0)
        >>> print(f"Found {len(outlier_cols)} outlier columns out of {weight.shape[1]}")
    """
    # For each input feature (column), check maximum absolute value
    col_max = weight.abs().max(dim=0).values  # [in_features]
    outlier_mask = col_max > threshold
    outlier_cols = outlier_mask.nonzero(as_tuple=True)[0].tolist()
    
    return outlier_cols


def split_weights_by_outliers(
    weight: torch.Tensor,
    outlier_cols: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Split weight matrix into normal (to be quantized) and outlier (keep FP16) columns.
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        outlier_cols: List of column indices to treat as outliers
    
    Returns:
        Tuple of:
        - weight_normal: Weights for normal columns [out_features, in_features - num_outliers]
        - weight_outlier: Weights for outlier columns [out_features, num_outliers]
        - normal_cols: List of normal column indices
        - outlier_cols: List of outlier column indices (same as input)
    
    Example:
        >>> weight = torch.randn(768, 768)
        >>> outlier_cols = [3, 47, 128]
        >>> w_normal, w_outlier, normal_idx, outlier_idx = split_weights_by_outliers(
        ...     weight, outlier_cols
        ... )
        >>> assert w_normal.shape == (768, 765)
        >>> assert w_outlier.shape == (768, 3)
    """
    in_features = weight.shape[1]
    
    # Get normal columns (everything except outliers)
    all_cols = set(range(in_features))
    normal_cols = sorted(list(all_cols - set(outlier_cols)))
    
    # Split weight matrix by columns
    weight_normal = weight[:, normal_cols]    # [out_features, in_features - num_outliers]
    weight_outlier = weight[:, outlier_cols]  # [out_features, num_outliers]
    
    return weight_normal, weight_outlier, normal_cols, outlier_cols


def separate_outliers(
    weight: torch.Tensor,
    threshold: float = 6.0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Convenience function to detect and split outliers in one call.
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        threshold: Absolute value threshold for outlier detection
    
    Returns:
        Same as split_weights_by_outliers()
    
    Example:
        >>> weight = torch.randn(768, 768)
        >>> w_normal, w_outlier, normal_idx, outlier_idx = separate_outliers(
        ...     weight, threshold=6.0
        ... )
    """
    outlier_cols = detect_outlier_columns(weight, threshold)
    return split_weights_by_outliers(weight, outlier_cols)

