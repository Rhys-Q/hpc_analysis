from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute import KeepPTX, KeepCUBIN

DEFAULT_BLOCK_SIZE = 128


# A, B, C are tensors on the GPU
@cute.kernel
def vector_add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < A.shape[0]:
        C[i] = A[i] + B[i]


@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
    block_size: cute.Uint32,
):
    grid_dim = [cute.ceil_div(N, block_size), 1, 1]
    vector_add(A, B, C).launch(grid=grid_dim, block=[block_size, 1, 1], smem=0)


def run(A, B, C, block_size: int = DEFAULT_BLOCK_SIZE):
    """
    Launch the cute kernel for vector add.
    """
    N = A.shape[0]
    solve(from_dlpack(A), from_dlpack(B), from_dlpack(C), N, block_size)


def emit_ptx(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
    block_size: cute.Uint32,
) -> Optional[str]:
    """
    Return PTX if the cute runtime exposes it; otherwise None.
    """
    vector_add = cute.compile[KeepPTX](
        solve, from_dlpack(A), from_dlpack(B), from_dlpack(C), N, block_size
    )
    return vector_add.__ptx__
