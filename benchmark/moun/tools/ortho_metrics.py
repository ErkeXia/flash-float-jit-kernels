# Copyright 2026 FlashFloat authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
from typing import Any, Dict, List, Optional

import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _fast_ortho_eval_kernel(
    X_ptr,
    Error_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Block-level matrix multiplication: A = X @ X^T
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        rk = k * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_k = rk[None, :] < N
        x1 = tl.load(
            X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xn,
            mask=(rm[:, None] < M) & mask_k,
            other=0.0,
        )
        x2 = tl.load(
            X_ptr + rn[:, None] * stride_xm + rk[None, :] * stride_xn,
            mask=(rn[:, None] < M) & mask_k,
            other=0.0,
        )

        acc += tl.dot(x1, tl.trans(x2))

    mask_m = rm[:, None] < M
    mask_n = rn[None, :] < M
    is_diag = rm[:, None] == rn[None, :]

    if False:
        # Compute squared Frobenius distance: (A - (1/M)*I)^2
        inv_M = 1.0 / M
        target = tl.where(is_diag, inv_M, 0.0)
        error_block = (acc - target) * (acc - target)
        error_block = tl.where(mask_m & mask_n, error_block, 0.0)

        tl.atomic_add(Error_ptr, tl.sum(error_block))
    else:
        # Compute squared Frobenius distance: (A - I)^2
        target = tl.where(is_diag, 1.0, 0.0)
        error_block = (acc - target) * (acc - target)
        error_block = tl.where(mask_m & mask_n, error_block, 0.0)
        tl.atomic_add(Error_ptr, tl.sum(error_block))


def fast_ortho_eval(X: torch.Tensor) -> float:
    """
    Evaluates the Frobenius-norm deviation of X from perfect conditioning.
    Assumes X will be scaled internally to stabilize spectral attributes.
    """
    if X.ndim > 2:
        X = X.flatten(1)
    if X.shape[0] > X.shape[1]:
        X = X.t()

    M, N = X.shape
    X_norm = X / (torch.norm(X, p="fro") + 1e-7)

    error_tensor = torch.zeros((1,), device=X.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_N))

    _fast_ortho_eval_kernel[grid](
        X_norm,
        error_tensor,
        M,
        N,
        X_norm.stride(0),
        X_norm.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return math.sqrt(error_tensor.item())


def gather_sharded_tensor(
    sharded_tensor: torch.Tensor,
    world_size: int,
    dst_rank: int = 0,
    detach: bool = True,
) -> Optional[torch.Tensor]:
    """
    Gathers a sharded tensor from all ranks into a full tensor on the destination rank.
    """
    if sharded_tensor is None:
        return None

    if world_size == 1:
        return sharded_tensor.detach() if detach else sharded_tensor

    sharded_tensor = sharded_tensor.contiguous()

    local_shape = sharded_tensor.shape
    full_shape = (local_shape[0] * world_size, *local_shape[1:])

    full_tensor = torch.empty(
        full_shape, dtype=sharded_tensor.dtype, device=sharded_tensor.device
    )

    dist.all_gather_into_tensor(full_tensor, sharded_tensor.contiguous())

    full_tensor = full_tensor.contiguous()

    # dist.barrier()
    # torch.cuda.synchronize()

    if dist.get_rank() == dst_rank:
        return full_tensor.detach() if detach else full_tensor
    return None


def get_sharded_slice(param: torch.nn.Parameter, p_cfg, rank: int):
    """
    Retrieves the appropriate slice of a parameter tensor based on sharding configuration of NorMuonAndAdam .
    """
    chunk_size = p_cfg.chunk_size
    start_idx = rank * chunk_size

    # 1. aquire data shard
    if p_cfg.comms.startswith("sharded"):
        # Jordan's GPT2 (trained with MuonAndAdam) specify reshape attributes for parameters updated by MuonAndAdam optimizer
        if hasattr(param, "reshape") and p_cfg.reshape is not None:
            # Sharded params use chunk state, chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])
            reshaped_data = param.data.view(p_cfg.reshape)
            data_slice = reshaped_data[start_idx : start_idx + chunk_size]
        else:
            data_slice = param.data[start_idx : start_idx + chunk_size]
    else:
        # replicated use full state
        data_slice = param.data

    # 2. aquire grad shard
    grad_slice = None
    if param.grad is not None:
        if p_cfg.comms.startswith("sharded"):
            if hasattr(param, "reshape") and p_cfg.reshape is not None:
                reshaped_grad = param.grad.view(p_cfg.reshape)
                grad_slice = reshaped_grad[start_idx : start_idx + chunk_size]
            else:
                grad_slice = param.grad[start_idx : start_idx + chunk_size]
        else:
            grad_slice = param.grad

    data_slice = data_slice.contiguous() if data_slice is not None else None
    grad_slice = grad_slice.contiguous() if grad_slice is not None else None
    return data_slice, grad_slice


def compute_condition_number(matrix):
    """Calculates the standard 2-norm condition number using SVD."""
    if matrix.dim() != 2:
        return float("nan")
    try:
        s = torch.linalg.svdvals(matrix.float())
        return (s[0] / (s[-1] + 1e-8)).item()
    except Exception:
        return float("nan")


def compute_frobenius_orth_error(X: torch.Tensor) -> float:
    """
    Computes the Frobenius norm of the deviation of X from an orthogonal matrix.
    """
    if X.dim() != 2:
        return float("nan")
    try:
        X_float = X.float()
        M = X_float.shape[0]
        if M > X_float.shape[1]:
            gram = X_float.T @ X_float
            identity = torch.eye(gram.shape[0], device=X.device)
        else:
            gram = X_float @ X_float.T
            identity = torch.eye(gram.shape[0], device=X.device)
        return torch.norm(gram - identity, p="fro").item()
    except Exception:
        return float("nan")


def compute_matrix_metrics(data, grad, momentum):
    """
    evluation metrics for 2-dimension matrix and 3-dimension
    Output:
        - cond: Condition Number for stability of the 2-dimensional matrix, computed via SVD
        - orth_error: frobenius ortho deviation error ||XX^T - I||_F
        - orth_dev: quick frobenius ortho deviation error with triton
    """
    metrics = {
        "data_cond": float("nan"),
        "data_orth_error": float("nan"),
        "data_orth_dev": float("nan"),
        "grad_cond": float("nan"),
        "grad_orth_error": float("nan"),
        "grad_orth_dev": float("nan"),
        "momentum_cond": float("nan"),
        "momentum_orth_error": float("nan"),
        "momentum_orth_dev": float("nan"),
    }

    def eval_bank(bank_tensor, name):
        conds, orth_errors, orth_devs = [], [], []

        if bank_tensor is None:
            return conds, orth_errors, orth_devs

        if bank_tensor.dim() == 3:
            for i in range(bank_tensor.shape[0]):
                matrix = bank_tensor[i]
                if matrix.dim() == 2 and matrix.numel() > 0:
                    try:
                        this_tensor = matrix.float()
                        if not (
                            torch.isnan(this_tensor).any()
                            or torch.isinf(this_tensor).any()
                        ):
                            conds.append(compute_condition_number(this_tensor))

                            orth_error = compute_frobenius_orth_error(this_tensor)
                            orth_errors.append(orth_error)

                            orth_devs.append(fast_ortho_eval(this_tensor))
                    except Exception as e:
                        print(f"Warn: fail to evaluate {name}[{i}] : {e}")
        elif bank_tensor.dim() == 2:
            try:
                this_tensor = bank_tensor.float()
                if not (
                    torch.isnan(this_tensor).any() or torch.isinf(this_tensor).any()
                ):
                    conds.append(compute_condition_number(this_tensor))
                    orth_errors.append(compute_frobenius_orth_error(this_tensor))
                    orth_devs.append(fast_ortho_eval(this_tensor))
            except Exception as e:
                print(f"Warn: fail to evaluate {name} : {e}")

        return conds, orth_errors, orth_devs

    def aggregate(name, conds, orth_errors, orth_devs):
        if conds:
            metrics[f"{name}_cond"] = sum(conds) / len(conds)
            metrics[f"{name}_cond_max"] = max(conds)  # 额外记录最病态的情况
            metrics[f"{name}_orth_error"] = sum(orth_errors) / len(orth_errors)
            metrics[f"{name}_orth_error_max"] = max(orth_errors)
            metrics[f"{name}_orth_dev"] = sum(orth_devs) / len(orth_devs)
            metrics[f"{name}_orth_dev_max"] = max(orth_devs)

    # main routine
    for tensor, name in [(data, "data"), (grad, "grad"), (momentum, "momentum")]:
        if tensor is not None and tensor.numel() > 0:
            conds, orth_errors, orth_devs = eval_bank(tensor, name)
            aggregate(name, conds, orth_errors, orth_devs)

    return metrics


def evaluate_condition_and_orthogonality(
    model,
    optimizer,
    step: int,
    prefix: str = "",
    dst_rank: int = 0,
    eval_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the condition number and orthogonality of Muon parameters in a distributed setting.
    """
    results = {}
    is_master = (dist.get_rank() == dst_rank) if dist.is_initialized() else True
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    normuon_params = [
        (param, p_cfg)
        for param, p_cfg in optimizer.param_cfgs.items()
        if p_cfg.optim == "normuon"
    ]

    if eval_params is not None:
        eval_set = set(eval_params)
        normuon_params = [
            (p, cfg) for (p, cfg) in normuon_params if cfg.label in eval_set
        ]

    if not normuon_params:
        if is_master:
            print(f"[{prefix}] step={step}: no Muon parameters found for evaluation.")
        return results

    for param, p_cfg in normuon_params:
        label = p_cfg.label

        # print(f"processing lable : {label}, cfg : {p_cfg} \n\n...")

        data_shard, grad_shard = get_sharded_slice(param, p_cfg, rank)

        momentum_shard = None
        if param in optimizer.param_states:
            p_state = optimizer.param_states[param]
            momentum_shard = p_state.get("momentum_buffer", None)

        full_data = (
            gather_sharded_tensor(data_shard, world_size, dst_rank)
            if data_shard is not None
            else None
        )

        full_grad = (
            gather_sharded_tensor(grad_shard, world_size, dst_rank)
            if grad_shard is not None
            else None
        )

        full_momentum = (
            gather_sharded_tensor(momentum_shard, world_size, dst_rank)
            if momentum_shard is not None
            else None
        )

        dist.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            print(f"[{prefix}] step={step} | {label}: gather sharded momentum done.")

        if is_master:
            metrics = compute_matrix_metrics(full_data, full_grad, full_momentum)
            results[label] = metrics

            print(f"[{prefix}] step={step} | {label}:")
            print(
                f"  Data:      cond={metrics['data_cond']:.2e}, orth_error={metrics['data_orth_error']:.6f}, orth_dev={metrics['data_orth_dev']:.6f}"
            )
            print(
                f"  Momentum:  cond={metrics['momentum_cond']:.2e}, orth_error={metrics['momentum_orth_error']:.6f}, orth_dev={metrics['momentum_orth_dev']:.6f}"
            )
            print(
                f"  Grad:      cond={metrics['grad_cond']:.2e}, orth_error={metrics['grad_orth_error']:.6f}, orth_dev={metrics['grad_orth_dev']:.6f}"
            )

        # del full_data, full_grad, full_momentum

        dist.barrier()
    return results


def test_condition_and_orthogonality():
    """Validates the adaptive Muon step-allocation logic."""
    print("=== Verification of Adaptive Muon Evaluation Logic ===")

    M, N = 128, 128
    threshold = 0.04
    print(f"Adaptive deviation threshold: {threshold}\n" + "-" * 60)

    # 1. Well-conditioned Matrix (QR decomposition with 1% noise)
    q, _ = torch.linalg.qr(torch.randn(M, N, device="cuda"))
    well_matrix = q + torch.randn(M, N, device="cuda") * 0.01

    # 2. Ill-conditioned Matrix (Explicitly decayed singular values)
    ill_matrix = torch.randn(M, N, device="cuda")
    u, _, v = torch.linalg.svd(ill_matrix, full_matrices=False)
    s_ill = torch.linspace(50.0, 0.01, steps=128, device="cuda")
    ill_matrix = u @ torch.diag(s_ill) @ v

    for name, mat in [
        ("Well-conditioned Matrix", well_matrix),
        ("Ill-conditioned Matrix", ill_matrix),
    ]:
        # SVD reference calculation (for profiling/validation use only)
        singular_values = torch.linalg.svdvals(mat)
        max_s = singular_values[0].item()
        min_s = singular_values[-1].item()
        condition_number = max_s / (min_s + 1e-8)

        def init_X0(G):
            X0 = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
            if G.size(-2) > G.size(-1):
                X0 = X0.mT
            return X0 / (X0.norm(dim=(-2, -1), keepdim=True) + 1e-7)

        def zeropower_via_newtonschulz5_adaptive(G, threshold=0.04):
            """Adaptive 5th-order Newton-Schulz iteration."""
            assert G.ndim >= 2
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()

            if G.size(-2) > G.size(-1):
                X = X.mT

            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
            ortho_dev_err_pre_ns = fast_ortho_eval(X)

            # Dynamic step allocation: 3 steps for stable matrices, 5 steps for ill-conditioned matrices
            steps = 3 if ortho_dev_err_pre_ns < threshold else 5

            for _ in range(steps):
                A = X @ X.mT
                B = b * A + c * A @ A
                X = a * X + B @ X

            if G.size(-2) > G.size(-1):
                X = X.mT

            ortho_dev_err_post_ns = fast_ortho_eval(X)
            return X, steps, ortho_dev_err_pre_ns, ortho_dev_err_post_ns

        X0 = init_X0(mat.clone())
        A0 = X0 @ X0.mT if mat.size(-2) <= mat.size(-1) else X0.mT @ X0
        init_identity_error = torch.norm(
            A0 - torch.eye(X0.shape[0], device="cuda"), p="fro"
        ).item()

        X, steps, pre_ns_ortho_dev_err, post_ns_ortho_dev_err = (
            zeropower_via_newtonschulz5_adaptive(mat.clone(), threshold=threshold)
        )
        X *= max(1, mat.size(-2) / mat.size(-1)) ** 0.5

        A = X @ X.mT if mat.size(-2) <= mat.size(-1) else X.mT @ X
        final_identity_error = torch.norm(
            A - torch.eye(X.shape[0], device="cuda"), p="fro"
        ).item()

        print(f"[{name}]:")
        print(
            f"  -> Analytical Condition Number (kappa): {condition_number:.2f} (Max SVD: {max_s:.2f}, Min SVD: {min_s:.2f})"
        )
        print(
            f"  -> Initial Orthogonal Error ||XX^T - I||_F: {init_identity_error:.6f}"
        )
        print(
            f"  -> Pre-NS Quick Orthogonal Deviation Error : {pre_ns_ortho_dev_err:.6f}"
        )
        print(f"  -> Allocated Newton-Schulz Steps: {steps}")
        print(
            f"  -> Final Orthogonal Error ||XX^T - I||_F ({steps} iters): {final_identity_error:.6f}"
        )
        print(
            f"  -> Post-NS Quick Orthogonal Deviation Error ({steps} iters): {post_ns_ortho_dev_err:.6f}"
        )
        print("-" * 60)


if __name__ == "__main__":
    test_condition_and_orthogonality()
