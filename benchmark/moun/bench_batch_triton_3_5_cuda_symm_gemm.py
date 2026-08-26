# Batch version of bench_triton_3_5_cuda_symm_gemm.py : batched (B, M, K) symmetric
# GEMM (X @ X.T) across all providers, aligned with the 2D benchmark:
#   - torch bf16 bmm baseline
#   - triton3_5 XXT reference
#   - gluon XXT reference
#   - ThunderMuon CUDA batched wasp_1p2c kernel (fp8 block-scaled)
#
# NOTE (yiakwy) : gluon correctness is intentionally NOT asserted here — numeric /
# host-side batch fixes land in another PR. Gluon is still benchmarked, looping over
# batch slices with the 2D path until its host-side batch view lands.
import itertools
import os
import time
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.testing

from jit_kernel.thunder_moun import symm_gemm_block_scaled
from jit_kernel.triton3_5.gluon.symm_gemm import GluonXXT

# TODO (yiakwy) : remove the reference
from jit_kernel.triton3_5.symm_gemm import XTX, XXT

SEED = 42

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


OCP_FP8E4M3_MAX = 448.0
FP8_MAX = OCP_FP8E4M3_MAX


DEBUG = False


@triton.jit
def fp8_weight_block_wise_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    N,
    SCALED_BLOCK_SIZE_M: tl.constexpr,
    SCALED_BLOCK_SIZE_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, SCALED_BLOCK_SIZE_N)
    offs_m = pid_m * SCALED_BLOCK_SIZE_M + tl.arange(0, SCALED_BLOCK_SIZE_M)
    offs_n = pid_n * SCALED_BLOCK_SIZE_N + tl.arange(0, SCALED_BLOCK_SIZE_N)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / FP8_MAX
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def fp8_weight_block_wise_quant(
    x: torch.Tensor, scaled_block_size_m: int = 128, scaled_block_size_n: int = 128
):
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        triton.cdiv(M, scaled_block_size_m),
        triton.cdiv(N, scaled_block_size_n),
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["SCALED_BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["SCALED_BLOCK_SIZE_N"]),
    )
    fp8_weight_block_wise_quant_kernel[grid](
        x,
        y,
        s,
        M,
        N,
        SCALED_BLOCK_SIZE_M=scaled_block_size_m,
        SCALED_BLOCK_SIZE_N=scaled_block_size_n,
        FP8_MAX=FP8_MAX,
    )
    return y, s


@triton.jit
def act_quant_kernel(
    x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, FP8_MAX: tl.constexpr
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / FP8_MAX
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, FP8_MAX=FP8_MAX)
    return y, s


def calculate_diff(b, m):
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # random batched pattern: every batch differs by construction, so a wrongly-folded
    # batch plane changes the values and gets caught by the checks below
    x = torch.randn(b, m, m, dtype=torch.float16, device="cuda")
    xs_0 = torch.ones((b, m, triton.cdiv(m, 128)), dtype=torch.float32, device="cuda")
    xs_1 = torch.ones(
        (b, triton.cdiv(m, 128), triton.cdiv(m, 128)), dtype=torch.float32, device="cuda"
    )

    # NOTE (yiakwy) : this can be computed asynchronously before Muon update
    # xq, xs_0 = act_quant(x)
    # wq, xs_1 = fp8_weight_block_wise_quant(x)

    xq_fp8 = x.to(torch.float8_e4m3fn)
    wq_fp8 = xq_fp8

    print("x.shape : ", x.shape)
    print("xs_0.shape : ", xs_0.shape)
    print("xs_1.shape : ", xs_1.shape)

    # fp32 reference, consistent with bench_triton_3_5_cuda_symm_gemm
    o_torch_ref = (x @ x.transpose(-1, -2)).cpu()

    def rel_err(a, ref):
        return (a - ref).norm().item() / ref.norm().item()

    x_bf16 = x.to(torch.bfloat16)

    o_triton_ref = XXT(x_bf16).cpu()
    e = rel_err(o_triton_ref, o_torch_ref)
    print(f"triton ref rel err (Frobenius) : {e:.4e}")
    assert e < 1e-2, f"triton ref mismatch : rel err {e:.4e}"

    o_symm_fp8_tril = symm_gemm_block_scaled(xq_fp8, wq_fp8, xs_0, xs_1).cpu()
    e = rel_err(o_symm_fp8_tril, o_torch_ref)
    print(f"cuda symm (batched) rel err : {e:.4e}")
    assert e < 5e-2, f"cuda symm (batched) mismatch : rel err {e:.4e}"

    print(f"✅ B={b} {m}x{m}x{m} batched thunder_moun_gemm")
    torch.cuda.synchronize()

    if DEBUG:
        return

    warmup = 25
    iters = 100
    for _ in range(warmup):
        _ = x @ x.transpose(-1, -2)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.time()
    start_event.record()
    for _ in range(iters):
        _ = x @ x.transpose(-1, -2)
    end_event.record()
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iters * 1000
    torch_device_elapsed = start_event.elapsed_time(end_event) / iters

    # === modded-gpt XXT - baseline (triton ref) ===
    for _ in range(warmup):
        _ = XXT(x_bf16)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.time()
    start_event.record()
    for _ in range(iters):
        _ = XXT(x_bf16)
    end_event.record()
    torch.cuda.synchronize()
    triton_ref_time = (time.time() - start) / iters * 1000
    triton_ref_device_elapsed = start_event.elapsed_time(end_event) / iters

    # === Gluon XXT - baseline  ===
    symm_gemm_op = GluonXXT()

    for _ in range(warmup):
        _ = symm_gemm_op(x_bf16)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.time()
    start_event.record()
    for _ in range(iters):
        _ = symm_gemm_op(x_bf16)
    end_event.record()
    torch.cuda.synchronize()

    gluon_ref_time = (time.time() - start) / iters * 1000
    gluon_ref_device_elapsed = start_event.elapsed_time(end_event) / iters

    # === ThunderMuon Symm Gemm - batched (CUDA) ===
    for _ in range(warmup):
        _ = symm_gemm_block_scaled(xq_fp8, wq_fp8, xs_0, xs_1)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.time()
    start_event.record()
    for _ in range(iters):
        o_symm_fp8_tril = symm_gemm_block_scaled(xq_fp8, wq_fp8, xs_0, xs_1)
    end_event.record()
    torch.cuda.synchronize()
    symm_gemm_time = (time.time() - start) / iters * 1000
    symm_gemm_device_elapsed = start_event.elapsed_time(end_event) / iters

    print(f"\nPerformance:")
    print(
        f"PyTorch                      : {torch_time:.3f} ms, device {torch_device_elapsed:.3f} ms"
    )
    print(
        f"Triton ref kernel            : {triton_ref_time:.3f} ms, device {triton_ref_device_elapsed:.3f} ms"
    )
    print(
        f"Gluon TMA kernel             : {gluon_ref_time:.3f} ms, device {gluon_ref_device_elapsed:.3f} ms"
    )
    print(
        f"Cuda Muon Symm Gemm (batched): {symm_gemm_time:.3f} ms, device {symm_gemm_device_elapsed:.3f} ms"
    )
    print(
        f"Speedup (cuda symm V3)       : {torch_time/symm_gemm_time:.2f}x, {torch_device_elapsed / symm_gemm_device_elapsed:.2f}x"
    )


B = [1, 4, 16]
M = [1024, 2048, 4096]


configs = list(itertools.product(B, M))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["b", "m"],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "torch",
            "triton_impl_ref",
            "gluon_muon_symm_gemm",
            "cuda_muon_symm_gemm",
        ],
        line_names=[
            "torch",
            "triton_ref",
            "gluon_muon_symm_gemm",
            "cuda_muon_symm_gemm",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
        ],
        ylabel="Latency",
        plot_name="muon-symm-gemm-batch-performance",
        args={},
    )
)
def benchmark(b: int, m: int, provider) -> None:
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x = torch.randn(b, m, m, dtype=torch.bfloat16, device="cuda")
    xs_0 = torch.ones((b, m, triton.cdiv(m, 128)), dtype=torch.float32, device="cuda")
    xs_1 = torch.ones(
        (b, triton.cdiv(m, 128), triton.cdiv(m, 128)), dtype=torch.float32, device="cuda"
    )

    # NOTE (yiakwy) : this can be computed asynchronously before Muon update
    # xq, xs_0 = act_quant(x)
    # wq, xs_1 = fp8_weight_block_wise_quant(x)

    x_fp8 = x.to(torch.float8_e4m3fn)
    wq_fp8 = x_fp8

    quantiles = [0.5, 0.2, 0.8]

    symm_gemm_op = GluonXXT()

    if provider == "torch":
        fn = lambda: x @ x.transpose(-1, -2)
    elif provider == "triton_impl_ref":
        fn = lambda: XXT(x)
    elif provider == "gluon_muon_symm_gemm":
        fn = lambda: symm_gemm_op(x)
    elif provider == "cuda_muon_symm_gemm":
        fn = lambda: symm_gemm_block_scaled(x_fp8, wq_fp8, xs_0, xs_1)

    # warm up
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    # Correctness check - simplified for CI
    if IS_CI:
        # Only test one configuration in CI
        test_configs = [configs[0]]
    else:
        test_configs = configs

    for cfg in test_configs:
        print(f"cfg : {cfg}")
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    if not DEBUG:
        print("Starting performance benchmark...")
        benchmark.run(print_data=True)
