from typing import Optional

import torch
import triton
import triton.language as tl

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


def run(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, block_size: int = DEFAULT_BLOCK_SIZE, num_warps: int = DEFAULT_NUM_WARPS, num_stages: int = DEFAULT_NUM_STAGES):
    """
    Launch the Triton vector add kernel.
    """
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


def emit_ptx(block_size: int = DEFAULT_BLOCK_SIZE, num_warps: int = DEFAULT_NUM_WARPS, num_stages: int = DEFAULT_NUM_STAGES, device: Optional[str] = None) -> str:
    """
    Compile the kernel and return PTX text.
    """
    binary = triton.compile(
        vector_add_kernel,
        signature="*fp32,*fp32,*fp32,i32",
        device=device or "cuda",
        constants={"BLOCK": block_size},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return binary.asm["ptx"]
