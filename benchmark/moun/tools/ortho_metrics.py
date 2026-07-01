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
import torch
import torch._dynamo as dynamo
import torch.distributed as dist

import triton
import triton.language as tl


@triton.jit
def _fast_ortho_eval_kernel(
    X_ptr, Error_ptr, 
    M, N, 
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
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
        x1 = tl.load(X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xn, mask=(rm[:, None] < M) & mask_k, other=0.0)
        x2 = tl.load(X_ptr + rn[:, None] * stride_xm + rk[None, :] * stride_xn, mask=(rn[:, None] < M) & mask_k, other=0.0)
        
        acc += tl.dot(x1, tl.trans(x2))
        
    mask_m = rm[:, None] < M
    mask_n = rn[None, :] < M
    is_diag = (rm[:, None] == rn[None, :])
    
    # Compute squared Frobenius distance: (A - (1/M)*I)^2
    inv_M = 1.0 / M
    target = tl.where(is_diag, inv_M, 0.0)
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
    X_norm = X / (torch.norm(X, p='fro') + 1e-7)
    
    error_tensor = torch.zeros((1,), device=X.device, dtype=torch.float32)
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_N))
    
    _fast_ortho_eval_kernel[grid](
        X_norm, error_tensor,
        M, N,
        X_norm.stride(0), X_norm.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    return math.sqrt(error_tensor.item())


def evaluate_condition_and_orthogonality(model, optimizer, step, prefix=""):
    """
    Tracks conditioning metrics and orthogonality for Muon-optimized parameters.
    """
    results = {}
    is_master = dist.get_rank() == 0 if dist.is_initialized() else True
    
    for param, p_cfg in optimizer.param_cfgs.items():
        if p_cfg.optim != "normuon":
            continue
            
        # Extract parameter slice depending on sharding configuration
        if p_cfg.comms.startswith("sharded"):
            rank = dist.get_rank() if dist.is_initialized() else 0
            if param.data.dim() >= 2:
                chunk_size = p_cfg.chunk_size
                start_idx = rank * chunk_size
                if hasattr(param, 'reshape') and p_cfg.reshape is not None:
                    reshaped = param.data.view(p_cfg.reshape)
                    data_to_eval = reshaped[start_idx:start_idx + chunk_size]
                else:
                    data_to_eval = param.data[start_idx:start_idx + chunk_size]
            else:
                data_to_eval = param.data
        else:
            data_to_eval = param.data
            
        # Extract gradients if available
        grad_to_eval = None
        if param.grad is not None:
            if p_cfg.comms.startswith("sharded"):
                if hasattr(param, 'reshape') and p_cfg.reshape is not None:
                    grad_reshaped = param.grad.view(p_cfg.reshape)
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    chunk_size = p_cfg.chunk_size
                    start_idx = rank * chunk_size
                    grad_to_eval = grad_reshaped[start_idx:start_idx + chunk_size]
                else:
                    grad_to_eval = param.grad
            else:
                grad_to_eval = param.grad
                
        # Extract momentum buffer states
        momentum_buffer = None
        if param in optimizer.param_states:
            p_state = optimizer.param_states[param]
            if 'momentum_buffer' in p_state:
                momentum_buffer = p_state['momentum_buffer']
                
        label = p_cfg.label
        metrics = compute_matrix_metrics(data_to_eval, grad_to_eval, momentum_buffer)
        results[label] = metrics
        
        if is_master:
            print(f"[{prefix}] step={step} | {label}: "
                  f"cond_data={metrics['data_cond']:.2e} "
                  f"cond_momentum={metrics['momentum_cond']:.2e} "
                  f"ortho_data={metrics['data_ortho']:.6f} "
                  f"ortho_momentum={metrics['momentum_ortho']:.6f}")
                  
    return results


def compute_matrix_metrics(data, grad, momentum):
    """Computes condition numbers and orthogonal deviations for a given state matrix."""
    metrics = {
        'data_cond': float('nan'), 'data_ortho': float('nan'),
        'grad_cond': float('nan'), 'grad_ortho': float('nan'),
        'momentum_cond': float('nan'), 'momentum_ortho': float('nan'),
    }
    
    if data is not None and data.dim() == 2 and data.numel() > 0:
        metrics['data_cond'] = compute_condition_number(data)
        metrics['data_ortho'] = fast_ortho_eval(data)
        
    if grad is not None and grad.dim() == 2 and grad.numel() > 0:
        metrics['grad_cond'] = compute_condition_number(grad)
        metrics['grad_ortho'] = fast_ortho_eval(grad)
        
    if momentum is not None and momentum.dim() == 2 and momentum.numel() > 0:
        metrics['momentum_cond'] = compute_condition_number(momentum)
        metrics['momentum_ortho'] = fast_ortho_eval(momentum)
        
    return metrics


def compute_condition_number(matrix):
    """Calculates the standard 2-norm condition number using SVD."""
    if matrix.dim() != 2:
        return float('nan')
    try:
        s = torch.linalg.svdvals(matrix.float())
        return (s[0] / (s[-1] + 1e-8)).item()
    except Exception:
        return float('nan')
    

def test_condition_and_orthogonality():
    """Validates the adaptive Muon step-allocation logic."""
    print("=== Verification of Adaptive Muon Evaluation Logic ===")
    
    M, N = 128, 128
    threshold = 0.04
    print(f"Adaptive deviation threshold: {threshold}\n" + "-"*60)

    # 1. Well-conditioned Matrix (QR decomposition with 1% noise)
    q, _ = torch.linalg.qr(torch.randn(M, N, device='cuda'))
    well_matrix = q + torch.randn(M, N, device='cuda') * 0.01 
    
    # 2. Ill-conditioned Matrix (Explicitly decayed singular values)
    ill_matrix = torch.randn(M, N, device='cuda')
    u, _, v = torch.linalg.svd(ill_matrix, full_matrices=False)
    s_ill = torch.linspace(50.0, 0.01, steps=128, device='cuda')
    ill_matrix = u @ torch.diag(s_ill) @ v
    
    for name, mat in [("Well-conditioned Matrix", well_matrix), ("Ill-conditioned Matrix", ill_matrix)]:
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
            ortho_err_0 = fast_ortho_eval(X)
            
            # Dynamic step allocation: 3 steps for stable matrices, 5 steps for ill-conditioned matrices
            steps = 3 if ortho_err_0 < threshold else 5
            
            for _ in range(steps):
                A = X @ X.mT
                B = b * A + c * A @ A
                X = a * X + B @ X
            
            if G.size(-2) > G.size(-1):
                X = X.mT

            ortho_err_1 = fast_ortho_eval(X)    
            return X, steps, ortho_err_0, ortho_err_1
        
        X0 = init_X0(mat.clone())
        A0 = X0 @ X0.mT if mat.size(-2) <= mat.size(-1) else X0.mT @ X0
        init_identity_error = torch.norm(A0 - torch.eye(X0.shape[0], device='cuda'), p='fro').item()

        X, steps, err, err_1 = zeropower_via_newtonschulz5_adaptive(mat.clone(), threshold=threshold)
        X *= max(1, mat.size(-2) / mat.size(-1))**0.5

        A = X @ X.mT if mat.size(-2) <= mat.size(-1) else X.mT @ X
        final_identity_error = torch.norm(A - torch.eye(X.shape[0], device='cuda'), p='fro').item()
        
        print(f"[{name}]:")
        print(f"  -> Analytical Condition Number (kappa): {condition_number:.2f} (Max SVD: {max_s:.2f}, Min SVD: {min_s:.2f})")
        print(f"  -> Initial Orthogonal Error ||XX^T - I||_F: {init_identity_error:.6f}")
        print(f"  -> Triton Scale Deviation Metric (dist to 1/M*I): {err:.6f}")
        print(f"  -> Allocated Newton-Schulz Steps: {steps}")
        print(f"  -> Final Orthogonal Error ||XX^T - I||_F ({steps} iters): {final_identity_error:.6f}")
        print(f"  -> Triton Final Scale Deviation Metric: {err_1:.6f}")
        print("-"*60)


if __name__ == "__main__":
    test_condition_and_orthogonality()