#!/usr/bin/env python3
"""
Utility module for path detection in cuBlade.
This module provides functions to find the repository root directory
and related paths dynamically across different machines.
"""

import os
from pathlib import Path

def find_repo_root(start_path=None):
    """
    Find the repository root by looking for specific directories.
    
    Args:
        start_path: Optional starting path for the search. If None, uses the directory of this script.
        
    Returns:
        str: Path to the repository root directory
    """
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
    
    # Convert to Path object for easier path manipulation
    current = Path(start_path)
    
    # Traverse up until we find a directory that looks like our repo root
    # The repo should have these key directories
    key_dirs = ["datasets", "models", "src"]
    
    # Start from current directory and go up to find repo root
    while current != current.parent:  # Stop at filesystem root
        # Check if this directory contains the key directories
        if all((current / d).exists() for d in key_dirs):
            return str(current)
        current = current.parent
    
    # If we didn't find it, use the directory containing the script
    print("Warning: Could not find repository root. Using script directory as base.")
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_base_paths():
    """
    Get all common base paths used in the project.
    
    Returns:
        dict: Dictionary of common paths
    """
    base_dir = find_repo_root()
        
    return {
            "KERNELS_PATH": Path(os.path.join(base_dir, "cublade", "kernels")),
            "BINDINGS_PATH": Path(os.path.join(base_dir, "cublade", "bindings")),
            "BASE_DIR": Path(base_dir),
        }

# if __name__ == "__main__":
#     # If run directly, print all paths
#     paths = get_base_paths()
#     print("DeepHistDLModule paths:")
#     for name, path in paths.items():
#         exists = os.path.exists(path)
#         status = "✓" if exists else "✗"
#         print(f"{name}: {path} [{status}]") 