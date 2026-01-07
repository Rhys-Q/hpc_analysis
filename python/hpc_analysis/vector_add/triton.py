from typing import Optional

import torch
import triton
import triton.language as tl
import os
import tempfile
from pathlib import Path
import shutil

DEFAULT_BLOCK_SIZE = 256
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2


@triton.jit
def vector_add_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)


def run(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N):
    """
    Launch the Triton vector add kernel.
    """
    block_size = DEFAULT_BLOCK_SIZE
    num_warps = DEFAULT_NUM_WARPS
    num_stages = DEFAULT_NUM_STAGES
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"
    assert A.shape == B.shape == C.shape
    N = A.numel()
    grid = (triton.cdiv(N, block_size),)
    vector_add_kernel[grid](
        A,
        B,
        C,
        N,
        BLOCK=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def emit_ptx(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N) -> str:
    """
    Compile the kernel and return PTX text.
    """
    # get env var
    cache_dir = os.environ.get("TRITON_CACHE_DIR", None)
    assert cache_dir is not None, "TRITON_CACHE_DIR must be set"

    # find the .ptx file from cache dir
    ptx_files = [str(p) for p in Path(cache_dir).rglob("*.ptx")]
    assert len(ptx_files) == 1, f"Expected 1 PTX file, got {len(ptx_files)}"
    ptx = Path(ptx_files[0]).read_text()
    return ptx
