from __future__ import annotations

import functools
import os
from typing import Optional

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = (
    "9.0a"  # Set the CUDA architecture to 9.0 for Hopper support
)

import warnings

import torch

USE_TORCH_JIT = False

# TODO (yaikwy) : use tvm ffi load_jit
if USE_TORCH_JIT:
    from torch.utils.cpp_extension import load as load_jit
else:
    # from jit_kernel.utils import load_jit
    from tvm_ffi.cpp import load, load_inline as load_jit

from jit_kernel.utils import KERNEL_PATH

extra_ldflags = ["-lcudart", "-lcuda"]

cuda_sources = [
    str(KERNEL_PATH / "csrc" / "thunder_moun/symm_1p2c_gemm.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_desc.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_desc_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_copy.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_copy_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_barrier.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_barrier.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/cluster/cluster.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/cluster/cluster.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/block.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/wasp_producer.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/sched.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/nv_block_1p2c_gemm_scaled_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/fragment/fragment.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/fragment/nv_frag_gemm_scaled_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/array_ref.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/layout.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/tuple.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/tensor_view_ref.h"),
]

_CPP_ENTRY = "symmetric_gemm_fp8_block_scaled"
_PY_SYMBOL = "symmetric_gemm_fp8_block_scaled"

common_cuda_flags = ["-O2", "-std=c++17"]


if torch.cuda.is_available():
    major, _minor = torch.cuda.get_device_capability()
else:
    major = 0

if major >= 9:
    #         "--ptxas-options=-v",
    common_cuda_flags += [
        "-O2",
        "-Xcompiler",
        "-fPIC",
        "-DENABLE_HOPPER=1",
        "-arch=compute_90a",
        "-code=sm_90a",
        "-I " + str(KERNEL_PATH / "csrc" / "thunder_moun"),
    ]


@functools.cache
def _jit_thunder_moun_module():
    if USE_TORCH_JIT:
        raise Exception(
            "Torch JIT is not supported for thunder moun kernel, please set USE_TORCH_JIT to False"
        )
    else:

        cuda_tmp_sources = [f'#include "{path}"' for path in cuda_sources]

        return load_jit(
            "thunder_moun",
            cuda_sources=cuda_tmp_sources,
            # TODO (yiakwy) : add sglang style TVM_FFI warpper
            # cuda_wrappers=[(_PY_SYMBOL, _CPP_ENTRY)],
            extra_cuda_cflags=common_cuda_flags,
            extra_ldflags=extra_ldflags,
        )


def is_column_major(tensor: torch.Tensor):
    return tensor.stride()[0] == 1 and tensor.stride()[1] > 1


BLK_K = 128


def symm_gemm_block_scaled(
    xq: torch.Tensor,
    wq: torch.Tensor,
    xs_lhs: torch.Tensor,
    xs_rhs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    SCALE_BLOCK_SIZE_N: int = 128,
    SCALE_BLOCK_SIZE_K: int = 128,
    check_input_shape: bool = False,
    use_mxfp8: bool = False,
) -> torch.Tensor:
    """
    Thunder Moun Optimizer CUDA Kernel
    Args:
        xq: input matrix of type torch[float16 | float8_e4m3fnuz | float8_e4m3] to perform xq * wq^T, of shape (M, K), by default row major
        wq: input matrix of type torch[float16 | float8_e4m3fnuz | float8_e4m3] to perform xq * wq^T, of shape (M, K), by default row major
        out: output maxtrix placeholder of torch[bfloat16] for x * x^T, of shape (M, M), by default row major
        xs_lhs: block scale of type torch[float16] for x, of shape (M, K // SCALE_BLOCK_SIZE_K), by default row major
        xs_rhs: block scale of type torch[float16] for x, of shape (N // SCALE_BLOCK_SIZE_N, K // SCALE_BLOCK_SIZE_K), by default row major
        use_fp8: use fp8 if True
    Returns:
        The output tensor of shape (M, M)
    """
    if check_input_shape:
        assert xq.dim() >= 2, "xq must be >= 2 dims"

        assert xq.is_cuda and wq.is_cuda, "xq/wq must be CUDA tensors"
        assert xq.shape[-1] == wq.shape[-1], "xq and wq must have the same K dimension"
        assert xq.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ), "xq must be FP8 e4m3"
        assert wq.dtype == xq.dtype, "wq must have the same dtype as xq"
        assert (
            xs_lhs.dtype == torch.float32 and xs_rhs.dtype == torch.float32
        ), "xs_lhs/xs_rhs must be float32 scales"

        if is_column_major(xq):
            warnings.warn(
                "xq is preferred to be in row major but found in column major in this algorithm"
            )

        if xq.dim() > 2:
            xq = xq.flatten(start_dim=1)

    M, K = xq.shape
    N = wq.shape[0]

    split_k = max(1, K // BLK_K)

    # NOTE (yiakwy) : disable on chip split_k reduction via NoC for the moment
    split_k = 1

    if out is None:
        out = torch.zeros((M, N), device=xq.device, dtype=torch.float16)

    module = _jit_thunder_moun_module()

    # print(f"X ptr: {xq.data_ptr()}, Aligned 16B? {xq.data_ptr() % 16 == 0}")
    # print(f"W ptr: {wq.data_ptr()}, Aligned 16B? {wq.data_ptr() % 16 == 0}")

    module.symmetric_gemm_fp8_block_scaled(
        xq,
        wq,
        xs_lhs,
        xs_rhs,
        out,
        M,
        N,
        K,
        xq.stride(0),
        xq.stride(1),
        wq.stride(0),
        wq.stride(1),
        out.stride(0),
        out.stride(1),
        split_k,
    )

    return out
