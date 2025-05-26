import torch
import torch.multiprocessing as mp
import triton
import triton.language as tl


def _cuda_info(info_queue):
    import pycuda.autoinit
    import pycuda.driver as drv
    from triton.runtime import driver

    properties = driver.utils.get_device_properties(0)
    attributes = drv.Device(0).get_attributes()
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = attributes[drv.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR]
    SIZE_SMEM = attributes[drv.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR]
    WARP_SIZE = attributes[drv.device_attribute.WARP_SIZE]

    info_queue.put({"NUM_SM": NUM_SM, "NUM_REGS": NUM_REGS, "SIZE_SMEM": SIZE_SMEM, "WARP_SIZE": WARP_SIZE})


_info_queue = mp.Queue()
mp.Process(target=_cuda_info, args=(_info_queue,)).start()
_cuinfo = _info_queue.get()
NUM_SM = _cuinfo["NUM_SM"]
NUM_REGS = _cuinfo["NUM_REGS"]
SIZE_SMEM = _cuinfo["SIZE_SMEM"]
WARP_SIZE = _cuinfo["WARP_SIZE"]


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    scale,
    causal,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    Z,
    H,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    barrier = n_cols

    for row_idx in range(row_start, n_rows, row_step):
        if causal:
            barrier = row_idx % M + 1

        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < barrier
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row *= scale
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


kernels = {}


def softmax(x, scale, causal):
    Z, H, M, N = x.shape
    n_rows = Z * H * M
    n_cols = N

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    x = x.view(n_rows, n_cols)
    y = y.view(n_rows, n_cols)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(
            y,
            x,
            scale,
            causal,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            Z,
            H,
            M,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.shared  # could be zero
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem) if size_smem else occupancy
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        scale,
        causal,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        Z,
        H,
        M,
        N,
    )
    return y.view(Z, H, M, N)
