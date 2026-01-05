"""
Shared utilities for benchmarking and artifact management.
"""

from .config import ARTIFACT_ROOT
from .io import ensure_dir, write_text
from .cuda import current_sm, have_cuda, require_torch_cuda
from .timing import measure_cuda
from .checks import assert_allclose
from .ptx import compile_ptx_to_sass, write_ptx_and_sass

__all__ = [
    "ARTIFACT_ROOT",
    "assert_allclose",
    "compile_ptx_to_sass",
    "current_sm",
    "ensure_dir",
    "have_cuda",
    "measure_cuda",
    "require_torch_cuda",
    "write_ptx_and_sass",
    "write_text",
]
