"""Run puzzles 9-11 with Triton-Viz's CPU interpreter."""

import torch
import torch.nn.functional as F
import triton_viz

from solutions_9_11 import conv2d_kernel, dot_kernel, flashatt_kernel


def launch(kernel, grid, *args, **meta):
    triton_viz.clear()
    triton_viz.trace("tracer")(kernel)[grid](*args, **meta)


def test_flash_attention():
    torch.manual_seed(0)
    q = torch.rand(200) - 0.5
    k = torch.rand(200) - 0.5
    v = torch.rand(200) - 0.5
    actual = torch.empty(200)

    launch(
        flashatt_kernel,
        (4, 1, 1),
        q,
        k,
        v,
        actual,
        N0=200,
        T=200,
        B0=64,
    )

    scores = q[:, None] * k[None, :]
    expected = (scores.softmax(dim=1) * v[None, :]).sum(dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_conv2d():
    torch.manual_seed(0)
    x = torch.rand(4, 8, 8) - 0.5
    kernel = torch.rand(4, 4) - 0.5
    actual = torch.empty(4, 8, 8)

    launch(
        conv2d_kernel,
        (4, 1, 1),
        x,
        kernel,
        actual,
        N0=4,
        H=8,
        W=8,
        KH=4,
        KW=4,
        B0=1,
    )

    padded = F.pad(x, (0, 4, 0, 4))
    expected = torch.empty_like(actual)
    for row in range(8):
        for col in range(8):
            expected[:, row, col] = (
                padded[:, row : row + 4, col : col + 4] * kernel
            ).sum(dim=(1, 2))
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_matrix_multiplication():
    torch.manual_seed(0)
    x = torch.rand(4, 32, 32) - 0.5
    y = torch.rand(4, 32, 32) - 0.5
    actual = torch.empty(4, 32, 32)

    launch(
        dot_kernel,
        (2, 2, 4),
        x,
        y,
        actual,
        N0=32,
        N1=32,
        N2=4,
        MID=32,
        B0=16,
        B1=16,
        B2=1,
        B_MID=16,
    )

    torch.testing.assert_close(actual, x @ y, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_flash_attention()
    print("Puzzle 9: PASS")
    test_conv2d()
    print("Puzzle 10: PASS")
    test_matrix_multiplication()
    print("Puzzle 11: PASS")
