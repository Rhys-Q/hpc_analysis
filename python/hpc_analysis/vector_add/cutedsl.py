import cutlass
import cutlass.cute as cute


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
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    ctx_cal_size = 128
    grid_dim = [cute.ceil_div(N, ctx_cal_size), 1, 1]
    vector_add(A, B, C).launch(grid=grid_dim, block=[ctx_cal_size, 1, 1], smem=0)
