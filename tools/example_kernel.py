"""
Example kernel for kernel_agent.py testing.

A simple vector add with two execution paths:
  * CUDA tensors -> real Triton JIT kernel launch (benchmarks real GPU perf)
  * CPU tensors  -> Triton CPU interpreter (enables harness runs without a GPU)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _launch(x, y, out, N, BLOCK_SIZE=1024):
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    if x.is_cuda:
        add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    else:
        from triton.runtime.interpreter import InterpretedFunction
        InterpretedFunction(add_kernel.fn).run(
            x, y, out, N, BLOCK_SIZE, grid=grid, warmup=False,
        )


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vector add — the function the agent optimizes."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {y.dtype}")

    N = x.numel()
    if N == 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)

    out = torch.empty(N, dtype=x.dtype, device=x.device)
    _launch(x.contiguous(), y.contiguous(), out, N)
    return out


def reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch reference — ground truth for harness correctness checks."""
    return torch.add(x, y)


def assert_correct(x, y, *, atol=1e-6, rtol=1e-6) -> torch.Tensor:
    """Run the kernel and assert it matches torch.add. Returns the output."""
    out = add(x, y)
    ref = reference(x, y)
    if not torch.allclose(out, ref, atol=atol, rtol=rtol):
        diff = (out - ref).abs()
        raise AssertionError(
            f"add_kernel mismatch: max_abs={diff.max().item():.3e} "
            f"mean_abs={diff.mean().item():.3e}"
        )
    return out


def self_test() -> None:
    """Value-level correctness harness."""
    torch.manual_seed(0)
    shapes = [(128,), (512,), (2048,), (63,), (4097,), (1023,), (1537,)]
    values = [0.0, 1e4, 1e-6]

    assert_correct(torch.randn(256), torch.randn(256))
    for shape in shapes:
        assert_correct(torch.randn(*shape), torch.randn(*shape))
    for v in values:
        assert_correct(torch.full((256,), v), torch.full((256,), v))
    x, y = torch.randn(4097), torch.randn(4097)
    o1 = add(x, y)
    o2 = add(x.clone(), y.clone())
    if not torch.allclose(o1, o2):
        raise AssertionError("determinism check failed")

    if torch.cuda.is_available():
        assert_correct(torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda"))
    print("self_test: ALL CHECKS PASSED")


if __name__ == "__main__":
    self_test()