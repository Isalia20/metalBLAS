from .dispatch import matmul, gemm
from . import kernels  # for cache priming

__all__ = ["matmul", "gemm", "kernels"]
