import ctypes
import hashlib
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from packaging import version

# using TVF FFI with triton
USE_TVM_FFI = True

tvm_ffi_modules = {
    "fp8_gemm_block_scaled": None,
    "thunder_moun_gemm": None,
}


if USE_TVM_FFI:
    import tvm_ffi

    from jit_kernel.triton3_4.tvm_ffi_mod import generate_tvm_ffi_source


def is_hopper():
    return triton.runtime.driver.active.get_current_target().arch >= 90


if is_hopper():
    # nvidia sm90 does not support chiplet architecture, defaults to 1 xcd per gpu
    NUM_XCDS = 1
    OCP_FP8E4M3_MAX = 448.0

    FP8_MAX = OCP_FP8E4M3_MAX

    props = torch.cuda.get_device_properties(0)
    SMs = props.multi_processor_count
    NUM_CUs = SMs
else:
    raise Exception("Not supported yet!")


class LoadBalanceStrategy(Enum):
    UNDEFINE = -1
    GAUSSIAN_FOLDING = 0


# NOTE (yiakwy) : useful for multiple-die chiplet architecture :
#   - Blackwell(2 dies), Rubin(2 dies),
#   - Rubin Ultra(4 dies), MI300(4 dies),
#   - MI300X(8 dies)
#
# Make sure SPLIT-K blocks cooperatively work on the same die to maximize the cache hit re-usage
@triton.jit
def _remmap_pid(tile_id, tall_xcds, BLOCKS_PER_XCD, NUM_XCDS):
    xcd = tile_id % NUM_XCDS
    local_pid = tile_id // NUM_XCDS  # (local_id, xcd), strides : (NUM_XCDS, 1)
    if xcd < tall_xcds:
        tile_id = (
            xcd * BLOCKS_PER_XCD + local_pid
        )  # (xcd, local_id), strides : (BLOCKS_PER_XCD, 1)
    else:
        tile_id = (
            tall_xcds * BLOCKS_PER_XCD
            + (xcd - tall_xcds) * (BLOCKS_PER_XCD - 1)
            + local_pid
        )
    return tile_id


# NOTE (yiakwy) : see our paper for cuda kernel for swizzle of lower left triangle
# pid = pid_m * (pid_m + 1) / 2 + pid_n
# pid_m*2 + pid_m - 2*pid - pid_n = 0
@triton.jit
def linear_to_tril(pid):
    # row = floor((sqrt(8*pid + 1) - 1) / 2)
    row = tl.floor((tl.math.sqrt(8.0 * pid + 1.0) - 1.0) / 2.0).to(tl.int32)
    col = pid - (row * (row + 1)) // 2
    return row, col


@triton.jit
def _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M):
    if GROUP_SIZE_M == 1:
        pid_n = tile_id % num_pid_n
        pid_m = tile_id // num_pid_n

        num_pid_in_group = num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _symm_moun_mat_compute(
    pid_m_pair,
    pid_m,
    pid_n,
    acc,
    acc_pair,
    xq_ptr,
    wq_ptr,
    xs_lhs_ptr,
    xs_rhs_ptr,
    o_ptr,
    M,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_wq_n,
    stride_wq_k,
    stride_xs_m,
    stride_xs_k,
    stride_ws_n,
    stride_ws_k,
    stride_o_m,
    stride_o_n,
    offs_k,
    start_ssteps,
    num_ssteps,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    SCALE_BLOCK_SIZE_N,
    SCALE_BLOCK_SIZE_K,
    SPLIT_K,
):
    # step 1 : comptue data offset
    offs_xq_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # if pid_m_pair >= pid_n:
    #     offs_xq_pair_m = (pid_m_pair * BLOCK_SIZE_M+ tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xq_pair_m = (pid_m_pair * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    if BLOCK_SIZE_M != BLOCK_SIZE_N or pid_m != pid_n:
        offs_wq_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % M
    else:
        offs_wq_n = offs_xq_m

    # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    xq_block_ptr = xq_ptr + (
        offs_xq_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
    )
    # if pid_m_pair >= pid_n:
    #     xq_pair_block_ptr = xq_ptr + (
    #         offs_xq_pair_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
    #     )
    xq_pair_block_ptr = xq_ptr + (
        offs_xq_pair_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
    )

    # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    # if BLOCK_SIZE_M != BLOCK_SIZE_N or pid_m != pid_n:
    #     wq_block_ptr = xq_ptr + (
    #         offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
    #     )
    # else:
    #     wq_block_ptr = xq_block_ptr
    wq_block_ptr = wq_ptr + (
        offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
    )

    # NOTE (yiakwy) : suppose BLOCK_SIZE_K <= SCALE_BLOCK_SIZE_K for simplicity
    # (BLOCK_SIZE_M, )
    xs_block_ptr = (
        xs_lhs_ptr
        + offs_xq_m * stride_xs_m
        + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    )
    # if pid_m_pair >= pid_n:
    #     xs_pair_block_ptr = (
    #         xs_lhs_ptr
    #         + offs_xq_pair_m * stride_xs_m
    #         + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    #     )
    xs_pair_block_ptr = (
        xs_lhs_ptr
        + offs_xq_pair_m * stride_xs_m
        + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    )

    offs_ws_n = offs_wq_n // SCALE_BLOCK_SIZE_N
    # (BLOCK_SIZE_N, )
    ws_block_ptr = (
        xs_rhs_ptr
        + offs_ws_n * stride_ws_n
        + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    )

    # step 2: pre-load data for the first iteration, note that we load one more block of data and scales to hide the compute latency by IO
    # for (pid_m, pid_n) = (m-1, 0), we load tile_x[m-1, k], tile_x[0, k]
    xq = tl.load(
        xq_block_ptr,
        mask=offs_k[None, :] < K,
        other=0.0,
    )

    xq_pair = xq
    if pid_m_pair >= pid_n:
        xq_pair = tl.load(
            xq_pair_block_ptr,
            mask=offs_k[None, :] < K,
            other=0.0,
        )

    # if BLOCK_SIZE_M != BLOCK_SIZE_N or pid_m != pid_n:
    #     wq = tl.load(
    #         wq_block_ptr,
    #         mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
    #         other=0.0,
    #     )
    # else:
    #     wq = xq
    wq = tl.load(
        wq_block_ptr,
        mask=offs_k[None, :] < K,
        other=0.0,
    )

    xs = tl.load(xs_block_ptr)

    xs_pair = xs
    if pid_n <= pid_m_pair:
        xs_pair = tl.load(xs_pair_block_ptr)

    ws = tl.load(ws_block_ptr)

    # NOTE (yiakwy) : Triton 3.4 hacks, allocate double buffer for the next block data and scales
    xq_next = xq
    wq_next = wq

    xs_next = xs
    ws_next = ws

    xq_pair_next = xq_pair
    xs_pair_next = xs_pair

    last_ssteps = start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    end_ssteps = start_ssteps + num_ssteps
    for k in tl.range(start_ssteps, end_ssteps):
        xq_k_step = BLOCK_SIZE_K * stride_xq_k
        wq_k_step = BLOCK_SIZE_K * stride_wq_k

        cur_ssteps = (k + 1) * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
        if k + 1 < end_ssteps:
            xq_next = tl.load(
                xq_block_ptr + xq_k_step,
                mask=offs_k[None, :] < K - (k + 1) * (BLOCK_SIZE_K),
                other=0.0,
            )

            # if BLOCK_SIZE_M != BLOCK_SIZE_N or pid_m != pid_n:
            #     wq_next = tl.load(
            #         wq_block_ptr + wq_k_step,
            #         mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
            #         other=0.0,
            #     )
            # else:
            #     wq_next = xq_next
            wq_next = tl.load(
                wq_block_ptr + wq_k_step,
                mask=offs_k[None, :] < K - (k + 1) * (BLOCK_SIZE_K),
                other=0.0,
            )

            if cur_ssteps > last_ssteps:
                xs_next = tl.load(xs_block_ptr + stride_xs_k)
                ws_next = tl.load(ws_block_ptr + stride_ws_k)

        # if pid_m == 1 and pid_n == 0:
        #     print("====== iteration k = ", k, " ======")
        #     print(f"acc before {k} iters: {acc}, xq = {xq}, wq = {wq}, xs = {xs}, ws = {ws}")

        # (xq * xs) @ (bq * bs).T = { cij = xs_i*ws_j* sum_k (xq_ik * wq_kj) }
        acc += tl.dot(xq, wq.trans(1, 0), input_precision="ieee") * (
            xs[:, None] * ws[None, :]
        )

        # if pid_m == 1 and pid_n == 0:
        #     print(f"acc after {k} iters : {acc}")
        #     pass

        if pid_m_pair >= pid_n:
            if k + 1 < end_ssteps:
                xq_pair_next = tl.load(
                    xq_pair_block_ptr + xq_k_step,
                    mask=offs_k[None, :] < K - (k + 1) * (BLOCK_SIZE_K),
                    other=0.0,
                )
                xs_pair_next = tl.load(xs_pair_block_ptr + stride_xs_k)

            acc_pair += tl.dot(xq_pair, wq.trans(1, 0), input_precision="ieee") * (
                xs_pair[:, None] * ws[None, :]
            )

        # advance pointer
        xq_block_ptr += xq_k_step
        wq_block_ptr += wq_k_step

        # advance scale pointer
        if cur_ssteps > last_ssteps:
            xs_block_ptr += stride_xs_k
            ws_block_ptr += stride_ws_k

        # move next data to current register
        if k + 1 < end_ssteps:
            xq = xq_next
            wq = wq_next

            if cur_ssteps > last_ssteps:
                xs = xs_next
                ws = ws_next

        # processing pair group together to maximize the data re-use in cache
        if pid_m_pair >= pid_n:
            xq_pair_block_ptr += xq_k_step

            if cur_ssteps > last_ssteps:
                xs_pair_block_ptr += stride_xs_k

            if k + 1 < end_ssteps:
                xq_pair = xq_pair_next

                if cur_ssteps > last_ssteps:
                    xs_pair = xs_pair_next

        last_ssteps = cur_ssteps

    # print(f"(pid_m, pid_n) = {pid_m, pid_n} : ", acc)
    if acc.dtype != o_ptr.type.element_ty:
        _acc = acc.to(o_ptr.type.element_ty)
    else:
        _acc = acc
    o_block_ptr = (
        o_ptr + offs_xq_m[:, None] * stride_o_m + offs_wq_n[None, :] * stride_o_n
    )
    o_mask = (offs_xq_m[:, None] < M) & (offs_wq_n[None, :] < M)
    if SPLIT_K == 1:
        tl.store(o_block_ptr, _acc, mask=o_mask)
    else:
        tl.atomic_add(o_block_ptr, _acc, mask=o_mask)

    # transpose copy for the symmetric tile
    if pid_n < pid_m:
        _acc = tl.trans(_acc, 1, 0)
        o_block_ptr = (
            o_ptr + offs_wq_n[:, None] * stride_o_m + offs_xq_m[None, :] * stride_o_n
        )
        o_mask = (offs_wq_n[:, None] < M) & (offs_xq_m[None, :] < M)
        if SPLIT_K == 1:
            tl.store(o_block_ptr, _acc, mask=o_mask)
        else:
            tl.atomic_add(o_block_ptr, _acc, mask=o_mask)

    # deal with pair
    # if pid_n <= pid_m_pair:
    #     print(f"(pid_m_pair, pid_n) = {pid_m_pair, pid_n} : ", acc_pair)

    if pid_n <= pid_m_pair:
        if acc_pair.dtype != o_ptr.type.element_ty:
            _acc = acc_pair.to(o_ptr.type.element_ty)
        else:
            _acc = acc_pair
        o_pair_block_ptr = (
            o_ptr
            + offs_xq_pair_m[:, None] * stride_o_m
            + offs_wq_n[None, :] * stride_o_n
        )
        o_mask_pair = (offs_xq_pair_m[:, None] < M) & (offs_wq_n[None, :] < M)
        if SPLIT_K == 1:
            tl.store(o_pair_block_ptr, _acc, mask=o_mask_pair)
        else:
            tl.atomic_add(o_pair_block_ptr, _acc, mask=o_mask_pair)

        # transpose copy for the symmetric tile
        if pid_n < pid_m_pair:
            _acc = tl.trans(_acc, 1, 0)
            o_pair_block_ptr = (
                o_ptr
                + offs_wq_n[:, None] * stride_o_m
                + offs_xq_pair_m[None, :] * stride_o_n
            )
            o_mask_pair = (offs_wq_n[:, None] < M) & (offs_xq_pair_m[None, :] < M)
            if SPLIT_K == 1:
                tl.store(o_pair_block_ptr, _acc, mask=o_mask_pair)
            else:
                tl.atomic_add(o_pair_block_ptr, _acc, mask=o_mask_pair)


# NOTE (yiakwy) : the implementation comes from flashFloat
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
            },
            num_stages=2,
            num_warps=8,
            maxnreg=384,
        ),
    ],
    key=("M", "K"),
    use_cuda_graph=True,
)
@triton.heuristics(
    {
        "NUM_PID_M": lambda args: triton.cdiv(args["M"] // 2, args["BLOCK_SIZE_M"]),
        "SPLIT_K": lambda args: args.get(
            "SPLIT_K", 1
        ),  # triton does not support NoC, see TLE (by Zhiyuan team) for replacement
    }
)
@triton.jit
def paired_symm_moun_mat_fp8_block_scaled_tuned_kernel(
    xq_ptr,
    wq_ptr,
    o_ptr,
    M,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_o_m,
    stride_o_n,
    xs_lhs_ptr,
    xs_rhs_ptr,
    stride_xs_lhs_m,
    stride_xs_lhs_k,
    stride_xs_rhs_n,
    stride_xs_rhs_k,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_CUs: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    M_EVEN: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    num_pid_m = NUM_PID_M

    if M_EVEN:
        FULL_BLOCKS = num_pid_m * 2 - 1
    else:
        FULL_BLOCKS = num_pid_m * 2

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_ssteps = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    start_ssteps = pid_k * num_ssteps
    offs_k += start_ssteps * BLOCK_SIZE_K

    # acc cross SPLIT-K & mirror jobs
    acc_dtype = tl.float32  # o_ptr.type.element_ty

    # non-persistent version to group pid_m and FULL_BLOCKS - pid_m together
    pid_m_pair = FULL_BLOCKS - pid_m
    for pid_n in tl.range(0, pid_m_pair + 1):
        # group (GROUP_SIZE=2) pid_m and M - pid_m together
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        acc_pair = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        _symm_moun_mat_compute(
            pid_m,
            pid_m_pair,
            pid_n,
            acc,
            acc_pair,
            xq_ptr,
            wq_ptr,
            xs_lhs_ptr,
            xs_rhs_ptr,
            o_ptr,
            M,
            K,
            stride_xq_m,
            stride_xq_k,
            stride_xq_m,
            stride_xq_k,
            stride_xs_lhs_m,
            stride_xs_lhs_k,
            stride_xs_rhs_n,
            stride_xs_rhs_k,
            stride_o_m,
            stride_o_n,
            offs_k,
            start_ssteps,
            num_ssteps,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_BLOCK_SIZE_N,
            SCALE_BLOCK_SIZE_K,
            SPLIT_K,
        )


# NOTE (yiakwy) : triton version implementation for tuning, not used in the final product
def thunder_moun_gemm(
    xq_lhs: torch.Tensor,
    xq_rhs: torch.Tensor,
    xs_0: torch.Tensor,
    xs_1: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    online_quant: bool = False,
    SPLIT_K: int = 1,
):
    M, K = xq_lhs.shape
    if out is None:
        out = torch.zeros((M, M), device=xq_lhs.device, dtype=torch.float16)

    # dipatch to kernel
    grid = lambda META: (SPLIT_K, triton.cdiv(M // 2, META["BLOCK_SIZE_M"]), 1)
    kernel = paired_symm_moun_mat_fp8_block_scaled_tuned_kernel

    if USE_TVM_FFI:

        def get_stable_kernel_key(M, K, **kwargs):
            key_str = f"M{M}_K{K}_" + "_".join(
                f"{k}{v}" for k, v in sorted(kwargs.items())
            )
            return hashlib.md5(key_str.encode("utf-8")).hexdigest()

        key = get_stable_kernel_key(M, K, SPLIT_K=SPLIT_K)

        if (
            tvm_ffi_modules["thunder_moun_gemm"] is not None
            and tvm_ffi_modules["thunder_moun_gemm"].get(key, None) is not None
        ):
            module, grid = tvm_ffi_modules["thunder_moun_gemm"][key]
            gx, gy, gz = grid

        else:
            # compile
            compiled_kernel = kernel[grid](
                xq_lhs,
                xq_rhs,
                out,
                M,
                K,
                xq_lhs.stride(0),
                xq_lhs.stride(1),
                out.stride(0),
                out.stride(1),
                xs_0,
                xs_1,
                xs_0.stride(0),
                xs_0.stride(1),
                xs_1.stride(0),
                xs_1.stride(1),
                SCALE_BLOCK_SIZE_K=128,
                SCALE_BLOCK_SIZE_N=128,
                NUM_XCDS=NUM_XCDS,
                NUM_CUs=NUM_CUs,
                SPLIT_K=SPLIT_K,
                M_EVEN=M % 2 == 0,
            )

            # make triton cubins
            cubin_bytes = compiled_kernel.kernel

            kernel_name = compiled_kernel.name
            sources, constants = generate_tvm_ffi_source(compiled_kernel, kernel_name)

            # register tvm ffi cubin launcher
            module = tvm_ffi.cpp.load_inline(
                "triton_loader",
                cuda_sources=sources,
                extra_ldflags=["-lcudart", "-lcuda"],
                embed_cubin={"triton_cubin": cubin_bytes},
            )

            gx = SPLIT_K
            gy = triton.cdiv(M // 2, constants["BLOCK_SIZE_M"][1])
            gz = 1

            print(f"[{kernel_name}] gx={gx}, gy={gy}, gz={gz}")

            # cache it
            if tvm_ffi_modules["thunder_moun_gemm"] is None:
                tvm_ffi_modules["thunder_moun_gemm"] = {}

            tvm_ffi_modules["thunder_moun_gemm"][key] = (module, (gx, gy, gz))

        # NOTE (yiakwy) : see cubin signature from generate_tvm_ffi_source, only leading dimensions (LD, i.e., **stride(0)** for row major) are kept as arguments, M, N, K are passed through constant memory, need to be consistent with the kernel definition
        # NOTE (yiakwy) : for alignment safty we convert all integers to int32_t in the cubin launcher, need to be consistent with the kernel definition as well
        module.paired_symm_moun_mat_fp8_block_scaled_tuned_kernel(
            xq_lhs,
            xq_rhs,
            out,
            int(M),
            int(K),
            int(xq_lhs.stride(0)),
            int(xq_lhs.stride(1)),
            int(out.stride(0)),
            int(out.stride(1)),
            xs_0,
            xs_1,
            int(xs_0.stride(0)),
            int(xs_0.stride(1)),
            int(xs_1.stride(0)),
            int(xs_1.stride(1)),
            int(gx),
            int(gy),
            int(gz),
        )
        return out
    else:
        kernel[grid](
            xq_lhs,
            xq_rhs,
            out,
            M,
            K,
            xq_lhs.stride(0),
            xq_lhs.stride(1),
            out.stride(0),
            out.stride(1),
            xs_0,
            xs_1,
            xs_0.stride(0),
            xs_0.stride(1),
            xs_1.stride(0),
            xs_1.stride(1),
            SCALE_BLOCK_SIZE_K=128,
            SCALE_BLOCK_SIZE_N=128,
            NUM_XCDS=NUM_XCDS,
            NUM_CUs=NUM_CUs,
            SPLIT_K=SPLIT_K,
            M_EVEN=M % 2 == 0,
        )

    return out


# NOTE (yiakwy) : triton normal block scale mm kernel for performance comparison, not used in the final product
def get_pack_size(
    pack_int8=False,
    pack_int16=False,
    pack_int32=False,
):
    if pack_int8:
        return 8
    if pack_int16:
        return 16
    if pack_int32:
        return 32
    return 1


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 16,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=16,
            maxnreg=384,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 16,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=8,
            maxnreg=384,
        ),
    ],
    key=("M", "N", "K"),
    use_cuda_graph=False,
)
@triton.heuristics(
    {
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
        "BLOCKS_PER_XCD": lambda args: triton.cdiv(
            triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
            * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
            NUM_XCDS,
        ),
        "NUM_PID_M": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"]),
        "NUM_PID_N": lambda args: triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
        "SPLIT_K": lambda _: 1,  # triton does not support NoC, SPLIT_K usually won't work unless the K dimension is extremely large
        "w1a16": lambda args: args.get("pack_int8", False)
        or args.get("pack_int16", False)
        or args.get(
            "pack_int32", False
        ),  # need to be consistent with the data layout of wq
        "packed_size": lambda args: get_pack_size(
            pack_int8=args.get("pack_int8", False),
            pack_int16=args.get("pack_int16", False),
            pack_int32=args.get("pack_int32", False),
        ),
        "_is_hopper": lambda args: args.get("_is_hopper", is_hopper()),
    }
)
@triton.jit
def fp8_streamk_gemm_block_scaled_tuned_kernel(
    xq_ptr,
    wq_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_wq_n,
    stride_wq_k,
    stride_o_m,
    stride_o_n,
    xs_ptr,
    ws_ptr,
    stride_xs_m,
    stride_xs_k,
    stride_ws_n,
    stride_ws_k,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_CUs: tl.constexpr,
    GRID_MN: tl.constexpr,
    BLOCKS_PER_XCD: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    w1a16: tl.constexpr,
    packed_size: tl.constexpr,
    use_w1a16_as_fp8: tl.constexpr,
    use_w1a8: tl.constexpr,
    _is_hopper: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = NUM_PID_M
    num_pid_n = NUM_PID_N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if w1a16:
        offs_k_packed = tl.arange(0, BLOCK_SIZE_K // packed_size)

    num_ssteps = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    for tile_id_0 in tl.range(pid, GRID_MN * SPLIT_K, NUM_CUs):
        tile_id = tile_id_0 % GRID_MN
        pid_k = tile_id_0 // GRID_MN

        start_ssteps = pid_k * num_ssteps
        offs_k += start_ssteps * BLOCK_SIZE_K
        if w1a16:
            offs_k_packed += start_ssteps * BLOCK_SIZE_K // packed_size

        pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

        offs_xq_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_wq_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        xq_block_ptr = xq_ptr + (
            offs_xq_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
        )
        if w1a16:
            # bits to unpack
            unpack_offs = tl.arange(0, packed_size).to(wq_ptr.type.element_ty)

            # (BLOCK_SIZE_N, BLOCK_SIZE_K // packed_size)
            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k_packed[None, :] * stride_wq_k
            )

        else:
            # (BLOCK_SIZE_N, BLOCK_SIZE_K)
            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
            )

            # (BLOCK_SIZE_M, )
            xs_block_ptr = (
                xs_ptr
                + offs_xq_m * stride_xs_m
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

            offs_ws_n = offs_wq_n // SCALE_BLOCK_SIZE_N
            # (BLOCK_SIZE_N, )
            ws_block_ptr = (
                ws_ptr
                + offs_ws_n * stride_ws_n
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

        acc_dtype = tl.float16  # tl.float32
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        if not w1a16:
            xs_next = tl.load(xs_block_ptr)
            ws_next = tl.load(ws_block_ptr)

        # NOTE (yiakwy) : different from
        last_ssteps = start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
        for k in tl.range(
            start_ssteps, start_ssteps + num_ssteps, num_stages=NUM_STAGES
        ):
            xq = tl.load(
                xq_block_ptr,
                mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                other=0.0,
            )

            if w1a16:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k_packed[None, :]
                    < K // packed_size - k * (BLOCK_SIZE_K // packed_size),
                    other=0.0,
                )

                # NOTE(yiakwy) : w1a16, unpack (BLOCK_SIZE_N, BLOCK_SIZE_K // pack_size) to (BLOCK_SIZE_N, pack_size, BLOCK_SIZE_K // pack_size)
                wq = wq[:, :, None]

                wq = tl.inline_asm_elementwise(
                    asm="bfe.u32 $0, $1, $2, 1;",
                    constraints="=r,r,r",
                    args=[wq, unpack_offs[None, None, :]],
                    dtype=wq_ptr.type.element_ty,
                    is_pure=True,
                    pack=1,
                )

                wq = wq.reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))
                wq = wq * 2 - 1
                wq = wq.to(tl.float16)

                if use_w1a16_as_fp8:
                    wq = wq.to(tl.float8e4nv)
                    xq = xq.to(tl.float8e4nv)
                elif use_w1a8:
                    wq = wq.to(tl.float8e4nv)

            else:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                    other=0.0,
                )

                if wq.dtype == tl.int8 or wq.dtype == tl.int16:
                    wq = wq.to(tl.float16)

                    if use_w1a16_as_fp8:
                        wq = wq.to(tl.float8e4nv)
                        xq = xq.to(tl.float8e4nv)
                    elif use_w1a8:
                        wq = wq.to(tl.float8e4nv)

            ws = ws_next
            xs = xs_next

            cur_ssteps = (k + 1) * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_next = tl.load(xs_block_ptr + stride_xs_k)
                ws_next = tl.load(ws_block_ptr + stride_ws_k)

            if w1a16:
                acc = tl.dot(xq, wq.trans(1, 0), acc=acc, out_dtype=acc_dtype)
            else:
                if False:  # _is_hopper:
                    # acc = tl.dot_scaled(xq, xs.reshape(BLOCK_SIZE_M, 1), "e4m3", wq.trans(1, 0), ws.reshape(BLOCK_SIZE_N, 1), "e4m3", acc, out_dtype=acc_dtype)
                    acc = tl.dot(
                        xq * xs[:, None],
                        (wq * ws[:, None]).trans(1, 0),
                        acc=acc,
                        input_precision="ieee",
                        out_dtype=acc_dtype,
                    )
                else:
                    # (xq * xs) @ (bq * bs).T = { cij = xs_i*ws_j* sum_k (xq_ik * wq_kj) }
                    acc += tl.dot(
                        xq, wq.trans(1, 0), input_precision="ieee", out_dtype=acc_dtype
                    ) * ((xs[:, None] * ws[None, :]).to(acc_dtype))

            xq_block_ptr += BLOCK_SIZE_K * stride_xq_k
            if w1a16:
                wq_block_ptr += BLOCK_SIZE_K // packed_size * stride_wq_k
            else:
                wq_block_ptr += BLOCK_SIZE_K * stride_wq_k

            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_block_ptr += stride_xs_k
                ws_block_ptr += stride_ws_k

            last_ssteps = cur_ssteps

        o_block_ptr = (
            o_ptr + offs_xq_m[:, None] * stride_o_m + offs_wq_n[None, :] * stride_o_n
        )
        o_mask = (offs_xq_m[:, None] < M) & (offs_wq_n[None, :] < N)

        if o_ptr.type.element_ty != acc.dtype:
            _acc = acc.to(o_ptr.type.element_ty)
        else:
            _acc = acc
        if SPLIT_K == 1:
            tl.store(o_block_ptr, _acc, mask=o_mask)
        else:
            tl.atomic_add(o_block_ptr, _acc, mask=o_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 16,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=8,
            maxnreg=384,
        ),
    ],
    key=("M", "N", "K"),
    use_cuda_graph=False,  # True,
)
@triton.heuristics(
    {
        "GRID_MN": lambda args: (
            triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
            * triton.cdiv(args["N"], args["BLOCK_SIZE_N"])
            + triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        )
        // 2,
        "BLOCKS_PER_XCD": lambda args: triton.cdiv(
            triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
            * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
            NUM_XCDS,
        ),
        "NUM_PID_M": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"]),
        "NUM_PID_N": lambda args: triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
        "SPLIT_K": lambda _: 1,  # triton does not support NoC, SPLIT_K usually won't work unless the K dimension is extremely large
        "w1a16": lambda args: args.get("pack_int8", False)
        or args.get("pack_int16", False)
        or args.get(
            "pack_int32", False
        ),  # need to be consistent with the data layout of wq
        "packed_size": lambda args: get_pack_size(
            pack_int8=args.get("pack_int8", False),
            pack_int16=args.get("pack_int16", False),
            pack_int32=args.get("pack_int32", False),
        ),
        "_is_hopper": lambda args: args.get("_is_hopper", is_hopper()),
        "_is_symm": lambda args: args.get("_is_symm", True),
    }
)
@triton.jit
def fp8_streamk_symm_gemm_block_scaled_tuned_kernel(
    xq_ptr,
    wq_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_wq_n,
    stride_wq_k,
    stride_o_m,
    stride_o_n,
    xs_ptr,
    ws_ptr,
    stride_xs_m,
    stride_xs_k,
    stride_ws_n,
    stride_ws_k,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_CUs: tl.constexpr,
    GRID_MN: tl.constexpr,
    BLOCKS_PER_XCD: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    w1a16: tl.constexpr,
    packed_size: tl.constexpr,
    use_w1a16_as_fp8: tl.constexpr,
    use_w1a8: tl.constexpr,
    _is_hopper: tl.constexpr,
    _is_symm: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = NUM_PID_M
    num_pid_n = NUM_PID_N

    tl.static_assert(
        _is_symm,
        "This streamk_symm_gemm kernel is designed for symmetric matrix multiplication, please set _is_symm to True.",
    )
    tl.static_assert(
        NUM_PID_M == NUM_PID_N,
        "The streamk_symm_gemm  kernel currently only support BLOCK_SIZE_M == BLOCK_SIZE_N.",
    )

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if w1a16:
        offs_k_packed = tl.arange(0, BLOCK_SIZE_K // packed_size)

    num_ssteps = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    for tile_id_0 in tl.range(pid, GRID_MN * SPLIT_K, NUM_CUs):
        tile_id = tile_id_0 % GRID_MN
        pid_k = tile_id_0 // GRID_MN

        start_ssteps = pid_k * num_ssteps
        offs_k += start_ssteps * BLOCK_SIZE_K
        if w1a16:
            offs_k_packed += start_ssteps * BLOCK_SIZE_K // packed_size

        if _is_symm:
            pid_m, pid_n = linear_to_tril(tile_id)
        else:
            pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

        offs_xq_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_wq_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        xq_block_ptr = xq_ptr + (
            offs_xq_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
        )
        if w1a16:
            # bits to unpack
            unpack_offs = tl.arange(0, packed_size).to(wq_ptr.type.element_ty)

            # (BLOCK_SIZE_N, BLOCK_SIZE_K // packed_size)
            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k_packed[None, :] * stride_wq_k
            )

        else:
            # (BLOCK_SIZE_N, BLOCK_SIZE_K)
            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
            )

            # (BLOCK_SIZE_M, )
            xs_block_ptr = (
                xs_ptr
                + offs_xq_m * stride_xs_m
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

            offs_ws_n = offs_wq_n // SCALE_BLOCK_SIZE_N
            # (BLOCK_SIZE_N, )
            ws_block_ptr = (
                ws_ptr
                + offs_ws_n * stride_ws_n
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

        acc_dtype = tl.float16  # tl.float32
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        if not w1a16:
            xs_next = tl.load(xs_block_ptr)
            ws_next = tl.load(ws_block_ptr)

        # NOTE (yiakwy) : different from
        last_ssteps = start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
        for k in tl.range(
            start_ssteps, start_ssteps + num_ssteps, num_stages=NUM_STAGES
        ):
            xq = tl.load(
                xq_block_ptr,
                mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                other=0.0,
            )

            if w1a16:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k_packed[None, :]
                    < K // packed_size - k * (BLOCK_SIZE_K // packed_size),
                    other=0.0,
                )

                # NOTE(yiakwy) : w1a16, unpack (BLOCK_SIZE_N, BLOCK_SIZE_K // pack_size) to (BLOCK_SIZE_N, pack_size, BLOCK_SIZE_K // pack_size)
                wq = wq[:, :, None]

                wq = tl.inline_asm_elementwise(
                    asm="bfe.u32 $0, $1, $2, 1;",
                    constraints="=r,r,r",
                    args=[wq, unpack_offs[None, None, :]],
                    dtype=wq_ptr.type.element_ty,
                    is_pure=True,
                    pack=1,
                )

                wq = wq.reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))
                wq = wq * 2 - 1
                wq = wq.to(tl.float16)

                if use_w1a16_as_fp8:
                    wq = wq.to(tl.float8e4nv)
                    xq = xq.to(tl.float8e4nv)
                elif use_w1a8:
                    wq = wq.to(tl.float8e4nv)

            else:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                    other=0.0,
                )

                if wq.dtype == tl.int8 or wq.dtype == tl.int16:
                    wq = wq.to(tl.float16)

                    if use_w1a16_as_fp8:
                        wq = wq.to(tl.float8e4nv)
                        xq = xq.to(tl.float8e4nv)
                    elif use_w1a8:
                        wq = wq.to(tl.float8e4nv)

            ws = ws_next
            xs = xs_next

            cur_ssteps = (k + 1) * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_next = tl.load(xs_block_ptr + stride_xs_k)
                ws_next = tl.load(ws_block_ptr + stride_ws_k)

            if w1a16:
                acc = tl.dot(xq, wq.trans(1, 0), acc=acc, out_dtype=acc_dtype)
            else:
                if False:  # _is_hopper:
                    # acc = tl.dot_scaled(xq, xs.reshape(BLOCK_SIZE_M, 1), "e4m3", wq.trans(1, 0), ws.reshape(BLOCK_SIZE_N, 1), "e4m3", acc, out_dtype=acc_dtype)
                    acc = tl.dot(
                        xq * xs[:, None],
                        (wq * ws[:, None]).trans(1, 0),
                        acc=acc,
                        input_precision="ieee",
                        out_dtype=acc_dtype,
                    )
                else:
                    # (xq * xs) @ (bq * bs).T = { cij = xs_i*ws_j* sum_k (xq_ik * wq_kj) }
                    acc += tl.dot(
                        xq, wq.trans(1, 0), input_precision="ieee", out_dtype=acc_dtype
                    ) * ((xs[:, None] * ws[None, :]).to(acc_dtype))

            xq_block_ptr += BLOCK_SIZE_K * stride_xq_k
            if w1a16:
                wq_block_ptr += BLOCK_SIZE_K // packed_size * stride_wq_k
            else:
                wq_block_ptr += BLOCK_SIZE_K * stride_wq_k

            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_block_ptr += stride_xs_k
                ws_block_ptr += stride_ws_k

            last_ssteps = cur_ssteps

        o_block_ptr = (
            o_ptr + offs_xq_m[:, None] * stride_o_m + offs_wq_n[None, :] * stride_o_n
        )
        o_mask = (offs_xq_m[:, None] < M) & (offs_wq_n[None, :] < N)

        if o_ptr.type.element_ty != acc.dtype:
            _acc = acc.to(o_ptr.type.element_ty)
        else:
            _acc = acc
        if SPLIT_K == 1:
            tl.store(o_block_ptr, _acc, mask=o_mask)
        else:
            tl.atomic_add(o_block_ptr, _acc, mask=o_mask)

        if _is_symm:
            # transpose copy for the symmetric tile
            if pid_n < pid_m:
                _acc = tl.trans(_acc, 1, 0)
                o_block_ptr = (
                    o_ptr
                    + offs_wq_n[:, None] * stride_o_m
                    + offs_xq_m[None, :] * stride_o_n
                )
                o_mask = (offs_wq_n[:, None] < M) & (offs_xq_m[None, :] < M)
                if SPLIT_K == 1:
                    tl.store(o_block_ptr, _acc, mask=o_mask)
                else:
                    tl.atomic_add(o_block_ptr, _acc, mask=o_mask)


def fp8_gemm_block_scaled(
    xq: torch.Tensor,
    wq: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    o: Optional[torch.Tensor] = None,
    SCALE_BLOCK_SIZE_N=128,
    SCALE_BLOCK_SIZE_K=128,
    check_input_shape=False,
    is_symm=False,
):
    if check_input_shape:
        assert xq.dim() >= 2, "xq must be >= 2 dims"
        assert wq.dim() >= 2, "wq must be >= 2 dims"

        if xq.dim() > 2:
            xq_flatten = xq.flatten(start_dim=1)
            wq_flatten = wq.flatten(start_dim=1)

            xq = xq_flatten
            wq = wq_flatten

        assert xq.shape[1] == wq.shape[1], "the shapes of xq and wq are incompatible!"

    M, K = xq.shape
    N = wq.shape[0]

    if o is None:
        o = torch.zeros((M, N), device=xq.device, dtype=torch.float16)

    # dispatch to kernel
    if is_symm:
        grid = lambda META: (
            min(
                NUM_CUs,
                (
                    (
                        triton.cdiv(M, META["BLOCK_SIZE_M"])
                        * triton.cdiv(N, META["BLOCK_SIZE_N"])
                        + triton.cdiv(M, META["BLOCK_SIZE_M"])
                    )
                    // 2
                )
                * META["SPLIT_K"],
            ),
        )

        kernel = fp8_streamk_symm_gemm_block_scaled_tuned_kernel
    else:
        grid = lambda META: (
            min(
                NUM_CUs,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"])
                * META["SPLIT_K"],
            ),
        )

        kernel = fp8_streamk_gemm_block_scaled_tuned_kernel

    if USE_TVM_FFI:
        global tvm_ffi_modules

        def get_stable_kernel_key(M, N, K, is_symm, **kwargs):
            key_str = f"M{M}_N{N}_K{K}_{1 if is_symm else 0}_" + "_".join(
                f"{k}{v}" for k, v in sorted(kwargs.items())
            )
            return hashlib.md5(key_str.encode("utf-8")).hexdigest()

        key = get_stable_kernel_key(M, N, K, is_symm, SPLIT_K=1)

        if (
            tvm_ffi_modules["fp8_gemm_block_scaled"] is not None
            and tvm_ffi_modules["fp8_gemm_block_scaled"].get(key, None) is not None
        ):
            module, grid = tvm_ffi_modules["fp8_gemm_block_scaled"][key]
            gx, gy, gz = grid
        else:
            # compile
            compiled_kernel = kernel[grid](
                xq,
                wq,
                o,
                M,
                N,
                K,
                xq.stride(0),
                xq.stride(1),
                wq.stride(0),
                wq.stride(1),
                o.stride(0),
                o.stride(1),
                xs,
                ws,
                xs.stride(0),
                xs.stride(1),
                ws.stride(0),
                ws.stride(1),
                SCALE_BLOCK_SIZE_N=SCALE_BLOCK_SIZE_N,
                SCALE_BLOCK_SIZE_K=SCALE_BLOCK_SIZE_K,
                NUM_XCDS=NUM_XCDS,
                NUM_CUs=NUM_CUs,
                use_w1a16_as_fp8=False,
                use_w1a8=False,
            )

            # make triton cubins
            cubin_bytes = compiled_kernel.kernel

            kernel_name = compiled_kernel.name
            sources, constants = generate_tvm_ffi_source(compiled_kernel, kernel_name)

            # register tvm ffi cubin launcher
            module = tvm_ffi.cpp.load_inline(
                "triton_loader",
                cuda_sources=sources,
                extra_ldflags=["-lcudart", "-lcuda"],
                embed_cubin={"triton_cubin": cubin_bytes},
            )

            if is_symm:
                gx = min(
                    NUM_CUs,
                    (
                        (
                            triton.cdiv(M, constants["BLOCK_SIZE_M"][1])
                            * triton.cdiv(N, constants["BLOCK_SIZE_N"][1])
                            + constants["BLOCK_SIZE_M"][1]
                        )
                        // 2
                    )
                    * constants["SPLIT_K"][1],
                )
            else:
                gx = min(
                    NUM_CUs,
                    triton.cdiv(M, constants["BLOCK_SIZE_M"][1])
                    * triton.cdiv(N, constants["BLOCK_SIZE_N"][1])
                    * constants["BLOCK_SIZE_K"][1],
                )
            gy = gz = 1

            print(f"[{kernel_name}] gx={gx}, gy={gy}, gz={gz}")

            # cache it
            if tvm_ffi_modules["fp8_gemm_block_scaled"] is None:
                tvm_ffi_modules["fp8_gemm_block_scaled"] = {}

            tvm_ffi_modules["fp8_gemm_block_scaled"][key] = (module, (gx, gy, gz))

        # NOTE (yiakwy) : see cubin signature from generate_tvm_ffi_source, only leading dimensions (LD, i.e., **stride(0)** for row major) are kept as arguments, M, N, K are passed through constant memory, need to be consistent with the kernel definition
        if is_symm:
            module.fp8_streamk_symm_gemm_block_scaled_tuned_kernel(
                xq,
                wq,
                o,
                int(M),
                int(N),
                int(K),
                int(xq.stride(0)),
                int(xq.stride(1)),
                int(wq.stride(0)),
                int(wq.stride(1)),
                int(o.stride(0)),
                int(o.stride(1)),
                xs,
                ws,
                int(xs.stride(0)),
                int(xs.stride(1)),
                int(ws.stride(0)),
                int(ws.stride(1)),
                int(gx),
                int(gy),
                int(gz),
            )
        else:
            module.fp8_streamk_gemm_block_scaled_tuned_kernel(
                xq,
                wq,
                o,
                int(M),
                int(N),
                int(K),
                int(xq.stride(0)),
                int(xq.stride(1)),
                int(wq.stride(0)),
                int(wq.stride(1)),
                int(o.stride(0)),
                int(o.stride(1)),
                xs,
                ws,
                int(xs.stride(0)),
                int(xs.stride(1)),
                int(ws.stride(0)),
                int(ws.stride(1)),
                int(gx),
                int(gy),
                int(gz),
            )
        return o
    else:
        kernel[grid](
            xq,
            wq,
            o,
            M,
            N,
            K,
            xq.stride(0),
            xq.stride(1),
            wq.stride(0),
            wq.stride(1),
            o.stride(0),
            o.stride(1),
            xs,
            ws,
            xs.stride(0),
            xs.stride(1),
            ws.stride(0),
            ws.stride(1),
            SCALE_BLOCK_SIZE_N=SCALE_BLOCK_SIZE_N,
            SCALE_BLOCK_SIZE_K=SCALE_BLOCK_SIZE_K,
            NUM_XCDS=NUM_XCDS,
            NUM_CUs=NUM_CUs,
            use_w1a16_as_fp8=False,
            use_w1a8=False,
        )

    return o
