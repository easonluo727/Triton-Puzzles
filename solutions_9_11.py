"""Solutions for Triton Puzzles 9, 10, and 11."""

import triton
import triton.language as tl


# Puzzle 9: Simple FlashAttention
@triton.jit
def flashatt_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr):
    pid = tl.program_id(0)
    B1: tl.constexpr = 32
    log2_e: tl.constexpr = 1.44269504

    q_offsets = pid * B0 + tl.arange(0, B0)
    q_mask = q_offsets < N0
    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

    running_max = tl.full((B0,), -1.0e6, tl.float32)
    running_sum = tl.zeros((B0,), tl.float32)
    accumulator = tl.zeros((B0,), tl.float32)

    for start in tl.range(0, T, B1):
        kv_offsets = start + tl.arange(0, B1)
        kv_mask = kv_offsets < T
        k = tl.load(k_ptr + kv_offsets, mask=kv_mask, other=0.0)
        scores = q[:, None] * k[None, :]
        scores = tl.where(q_mask[:, None] & kv_mask[None, :], scores, -1.0e6)

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp2((running_max - new_max) * log2_e)
        probabilities = tl.exp2((scores - new_max[:, None]) * log2_e)

        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        v = tl.load(v_ptr + kv_offsets, mask=kv_mask, other=0.0)
        accumulator = accumulator * correction + tl.sum(
            probabilities * v[None, :], axis=1
        )
        running_max = new_max

    output = accumulator / running_sum
    tl.store(z_ptr + q_offsets, output, mask=q_mask)


# Puzzle 10: Two Dimensional Convolution
@triton.jit
def conv2d_kernel(
    x_ptr,
    k_ptr,
    z_ptr,
    N0,
    H,
    W,
    KH: tl.constexpr,
    KW: tl.constexpr,
    B0: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    batch_offsets = pid_0 * B0 + tl.arange(0, B0)
    batch_mask = batch_offsets < N0

    kernel_rows = tl.arange(0, KH)
    kernel_cols = tl.arange(0, KW)
    kernel_offsets = kernel_rows[:, None] * KW + kernel_cols[None, :]
    kernel = tl.load(k_ptr + kernel_offsets)

    for output_row in tl.range(0, H):
        for output_col in tl.range(0, W):
            input_rows = output_row + kernel_rows[None, :, None]
            input_cols = output_col + kernel_cols[None, None, :]
            input_offsets = (
                batch_offsets[:, None, None] * H * W
                + input_rows * W
                + input_cols
            )
            input_mask = (
                batch_mask[:, None, None]
                & (input_rows < H)
                & (input_cols < W)
            )
            window = tl.load(
                x_ptr + input_offsets, mask=input_mask, other=0.0
            )
            result = tl.sum(
                tl.sum(window * kernel[None, :, :], axis=2), axis=1
            )

            output_offsets = (
                batch_offsets * H * W + output_row * W + output_col
            )
            tl.store(z_ptr + output_offsets, result, mask=batch_mask)


# Puzzle 11: Matrix Multiplication
@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)

    batch_offsets = pid_2 * B2 + tl.arange(0, B2)
    row_offsets = pid_0 * B0 + tl.arange(0, B0)
    col_offsets = pid_1 * B1 + tl.arange(0, B1)
    batch_mask = batch_offsets < N2
    row_mask = row_offsets < N0
    col_mask = col_offsets < N1

    accumulator = tl.zeros((B2, B0, B1), tl.float32)

    for start in tl.range(0, MID, B_MID):
        mid_offsets = start + tl.arange(0, B_MID)
        mid_mask = mid_offsets < MID

        x_offsets = (
            batch_offsets[:, None, None] * N0 * MID
            + row_offsets[None, :, None] * MID
            + mid_offsets[None, None, :]
        )
        y_offsets = (
            batch_offsets[:, None, None] * MID * N1
            + mid_offsets[None, :, None] * N1
            + col_offsets[None, None, :]
        )
        x_mask = (
            batch_mask[:, None, None]
            & row_mask[None, :, None]
            & mid_mask[None, None, :]
        )
        y_mask = (
            batch_mask[:, None, None]
            & mid_mask[None, :, None]
            & col_mask[None, None, :]
        )
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        y = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
        accumulator += tl.dot(x, y)

    output_offsets = (
        batch_offsets[:, None, None] * N0 * N1
        + row_offsets[None, :, None] * N1
        + col_offsets[None, None, :]
    )
    output_mask = (
        batch_mask[:, None, None]
        & row_mask[None, :, None]
        & col_mask[None, None, :]
    )
    tl.store(z_ptr + output_offsets, accumulator, mask=output_mask)
