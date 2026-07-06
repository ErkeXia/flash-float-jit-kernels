import copy
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

dynamo.config.recompile_limit = 64

# must set to be False to avoid torch._dynamo error in distributed test
torch._dynamo.config.disable = False

## GPT2 training script
world_size = 2
rank = 0
grad_accum_steps = 1


## Mock of GPT2 TrainingSchedule
@dataclass
class TrainingSchedule:
    split_step: int = 10
    total_steps: int = 20

    def lookup(self, step):
        return None, 0.0

    def get_lr(self, step):
        return 1.0

    def get_transition_steps(self):
        return []


training_schedule = TrainingSchedule()


def get_muon_momentum(step):
    return 0.95


@dataclass
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor = None
    ws_short: int = 128
    ws_long: int = 384
    train_max_seq_len: int = 2048


## Mock GPT2 model trained by MuonAndAdam
def create_mock_param_table() -> Dict:
    """Mock TrainingManager.param_table"""
    return {
        "qk_bank": {"optim": "normuon", "comms": "sharded", "adam_betas": None},
        "vo_bank": {"optim": "normuon", "comms": "sharded", "adam_betas": None},
        "mlp_bank": {"optim": "normuon", "comms": "sharded", "adam_betas": None},
        "scalars": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
            "lr_mul": 5.0,
            "wd_mul": 0.0,
        },
        "smear_gate": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
            "lr_mul": 0.01,
            "wd_mul": 0.0,
        },
        "skip_gate": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
            "lr_mul": 0.05,
            "wd_mul": 0.0,
        },
        "attn_gate_bank": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
        },
        "ve_gate_bank": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
        },
        "lm_head": {
            "optim": "adam",
            "comms": "sharded",
            "adam_betas": [0.5, 0.95],
            "wd_mul": 150.0,
        },
        "embed": {
            "optim": "adam",
            "comms": "sharded",
            "adam_betas": [0.5, 0.95],
            "wd_mul": 150.0,
        },
        "post_lambdas": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 1.0,
            "wd_mul": 0.0,
        },
        "x0_lambdas": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 1.0,
            "wd_mul": 0.0,
        },
        "bigram_lambdas": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 1.0,
            "wd_mul": 0.0,
        },
        "resid_lambdas": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 5.0,
            "wd_mul": 0.0,
        },
        "xsa_alphas": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 1.0,
            "wd_mul": 0.0,
        },
        "value_embeds": {
            "optim": "adam",
            "comms": "sharded",
            "adam_betas": [0.75, 0.95],
            "lr_mul": 75.0,
            "wd_mul": 5.0,
        },
        "mudd_w1": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
            "lr_mul": 0.25,
        },
        "mudd_w2": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
            "lr_mul": 0.25,
        },
        "mudd_b2": {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.99],
            "lr_mul": 0.25,
            "wd_mul": 0.0,
        },
    }


def create_mock_model():
    """Mock GPT2 model with parameters matching the param_table"""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Parameters for Muon (sharded)
            self.qk_bank = nn.Parameter(torch.randn(64, 256, 768) * 0.1)
            self.qk_bank.reshape_tuple = (64, 256, 768)
            self.qk_bank.reshape = self.qk_bank.reshape_tuple

            self.vo_bank = nn.Parameter(torch.randn(24, 768, 768) * 0.1)
            self.vo_bank.reshape_tuple = (24, 768, 768)
            self.vo_bank.reshape = self.vo_bank.reshape_tuple

            self.mlp_bank = nn.Parameter(torch.randn(12, 2, 3072, 768) * 0.1)
            self.mlp_bank.reshape_tuple = (24, 3072, 768)
            self.mlp_bank.reshape = self.mlp_bank.reshape_tuple

            # Parameters for Adam (replicated)
            self.scalars = nn.Parameter(torch.randn(10) * 0.1)

            self.smear_gate = nn.Parameter(torch.randn(12, 1) * 0.01)

            self.skip_gate = nn.Parameter(torch.randn(12, 1) * 0.01)

            self.attn_gate_bank = nn.Parameter(torch.randn(10, 6, 12) * 0.1)

            self.ve_gate_bank = nn.Parameter(torch.randn(5, 6, 12) * 0.1)

            self.post_lambdas = nn.Parameter(torch.randn(11, 2) * 0.1)

            self.x0_lambdas = nn.Parameter(torch.randn(11) * 0.1)

            self.bigram_lambdas = nn.Parameter(torch.randn(11) * 0.1)

            self.resid_lambdas = nn.Parameter(torch.randn(11, 2) * 0.1)

            self.value_embeds = nn.Parameter(torch.randn(5 * 50304, 768) * 0.1)

            self.mudd_w1 = nn.Parameter(torch.randn(2, 64, 768) * 0.1)

            self.mudd_w2 = nn.Parameter(torch.randn(2, 14, 64) * 0.1)

            self.mudd_b2 = nn.Parameter(torch.randn(2, 14) * 0.1)

            # Parameters for Adam (sharded)
            self.lm_head = nn.Parameter(torch.randn(768, 50304) * 0.1)

            self.embed = nn.Parameter(torch.randn(50304, 768) * 0.1)

            self.xsa_alphas = nn.Parameter(torch.randn(11, 6) * 0.1)

            self.yarn = None

            self.yarn_paired_head = None

        def named_parameters(self, prefix: str = "", recurse: bool = True):
            for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
                param.label = name.replace(".weight", "").replace(".bias", "")
                yield name, param

    return MockModel()


# Mock training manager
class MockTrainingManager:
    """
    模拟 TrainingManager，用于单测环境
    包含完整的 param_table、optimizer 初始化和 step_optimizers 逻辑
    """

    def __init__(self, model, world_size: int):
        self.model = model
        self.world_size = world_size
        self.param_table = create_mock_param_table()

        # ordered from small to large, process small values first, then large values
        self.work_order = [
            "scalars",
            "smear_gate",
            "skip_gate",
            "attn_gate_bank",
            "ve_gate_bank",
            "mudd_b2",
            "xsa_alphas",
            "post_lambdas",
            "x0_lambdas",
            "bigram_lambdas",
            "resid_lambdas",
        ] + [
            "mudd_w2",
            "value_embeds",
            "mudd_w1",
            "lm_head",
            "embed",
            "qk_bank",
            "vo_bank",
            "mlp_bank",
        ]

        adam_defaults = dict(lr=0.008, eps=1e-10, weight_decay=0.005)
        normuon_defaults = dict(lr=0.023, momentum=0.95, beta2=0.9, weight_decay=1.2)

        from nor_muon_adam import NorMuonAndAdam

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=list(self.param_table),
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
        )

        self.split_step = 10

    def step_optimizers(self, step: int):
        step_lr = 1.0
        muon_momentum = 0.95
        do_adam = step % 2 == 1

        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr
            if p_cfg.optim == "normuon":
                p_cfg.momentum = muon_momentum

        self.optimizer.step(do_adam=do_adam)


# Main Test
def test_training_manager_orthogonality(rank: int, world_size: int):
    """
    Verify that evaluate_condition_and_orthogonality can capture NorMuon parameters, grads, and momentum buffers
    correctly before and after step_optimizers in a distributed setting.
    """

    # init distributed environment
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world_size
    )

    is_master = rank == 0

    print(
        f"Rank {rank}: dist env is intialized with world_size={world_size}, is_master={is_master}"
    )

    model = create_mock_model().cuda()

    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()

    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.qk_bank.data = model.qk_bank.data.bfloat16()
    model.vo_bank.data = model.vo_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()
    model.mudd_w1.data = model.mudd_w1.data.bfloat16()
    model.mudd_w2.data = model.mudd_w2.data.bfloat16()
    model.mudd_b2.data = model.mudd_b2.data.bfloat16()

    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    manager = MockTrainingManager(model, world_size)

    def create_ill_matrix(M, N):
        u, _, v = torch.linalg.svd(torch.randn(M, N, device="cuda"))
        s_ill = torch.linspace(100.0, 0.01, steps=min(M, N), device="cuda")
        s_full = torch.zeros(min(M, N), device="cuda")
        s_full[: len(s_ill)] = s_ill
        s_full[len(s_ill) :] = 1e-3

        return u, s_full, s_ill, v

    # mock grad and momentum buffer for normuon parameters
    for param, p_cfg in manager.optimizer.param_cfgs.items():
        p_state = manager.optimizer.param_states[param]

        if param.grad is None:
            if hasattr(param, "reshape_tuple"):
                flat_grad = (
                    torch.randn(
                        param.reshape_tuple, device=param.device, dtype=param.dtype
                    )
                    * 0.1
                )
                param.grad = flat_grad.view(param.shape)
            else:
                param.grad = torch.randn_like(param) * 0.1

        if "momentum_buffer" in p_state:
            if p_cfg.label == "mlp_bank":
                buf_shape = p_state["momentum_buffer"].shape  # [12, 3072, 768]

                if len(buf_shape) == 3:
                    B, M, N = buf_shape
                    for i in range(B):
                        u, s_full, s_ill, v = create_ill_matrix(M, N)
                        p_state["momentum_buffer"][i] = (
                            u[:, : min(M, N)] @ torch.diag(s_full) @ v[: min(M, N), :]
                        )

                    print(
                        f"✅ Rank {rank}: {p_cfg.label} construct 3-D ill-conditioned momentum buffer with shape {buf_shape}"
                    )
                else:
                    u, _, v = torch.linalg.svd(
                        torch.randn(buf_shape[0], buf_shape[1], device="cuda")
                    )
                    s_ill = torch.linspace(
                        100.0, 0.01, steps=buf_shape[0], device="cuda"
                    )
                    s_full = torch.cat(
                        [
                            s_ill,
                            torch.ones(buf_shape[0] - len(s_ill), device="cuda") * 1e-3,
                        ]
                    )

                    p_state["momentum_buffer"] = (
                        u[:, : buf_shape[0]] @ torch.diag(s_full) @ v[: buf_shape[0]]
                    )

                    print(
                        f"✅ Rank {rank}: {p_cfg.label} construct 2-D ill-conditioned momentum buffer with shape {buf_shape}"
                    )
            else:
                p_state["momentum_buffer"] = (
                    torch.randn_like(p_state["momentum_buffer"]) * 0.1
                )

        if "momentum_buffer" in p_state:
            p_state["mantissa"] = torch.zeros(
                p_state["momentum_buffer"].shape,
                dtype=torch.uint16,
                device=p_state["momentum_buffer"].device,
            )
            print(
                f"Rank {rank}: initialize mantissa fo {p_cfg.label} with shape {p_state['mantissa'].shape}"
            )

    dist.barrier()

    if is_master:
        print("\n" + "=" * 80)
        print("pre step_optimizers evaluation (pre-step)")
        print("=" * 80)

    from ortho_metrics import evaluate_condition_and_orthogonality

    pre_results = evaluate_condition_and_orthogonality(
        model, manager.optimizer, step=1, prefix="pre_step"
    )

    if is_master:
        print("\n执行 step_optimizers (step=1)...")
    manager.step_optimizers(step=1)
    dist.barrier()

    if is_master:
        print("\n" + "=" * 80)
        print("post step_optimizers evaluation (post-step)")
        print("=" * 80)

    post_results = evaluate_condition_and_orthogonality(
        model, manager.optimizer, step=1, prefix="post_step"
    )

    if is_master:
        print("\n" + "=" * 80)
        print("Result :")
        print("=" * 80)

        expected = [
            k for k, v in manager.param_table.items() if v["optim"] == "normuon"
        ]
        actual = list(pre_results.keys())
        print(f"real eval: {expected}")
        print(f"actual eval: {actual}")
        assert set(expected) == set(
            actual
        ), f"different parameters: expected {expected}, actual {actual}"

        for label in expected:
            print(f"\n[{label}] pre-post:")
            pre = pre_results[label]
            post = post_results[label]

            print(
                f"  Pre-step:  data_cond={pre['data_cond']:.2e}, momentum_cond={pre['momentum_cond']:.2e}"
            )
            print(
                f"  Post-step: data_cond={post['data_cond']:.2e}, momentum_cond={post['momentum_cond']:.2e}"
            )

            assert pre["data_cond"] >= 1.0 and post["data_cond"] >= 1.0
            assert pre["momentum_cond"] >= 1.0 and post["momentum_cond"] >= 1.0
            assert pre["data_orth_dev"] >= 0 and post["data_orth_dev"] >= 0

        # verify momentum of mlp_bank is ill-conditioned (cond > 100)
        if "mlp_bank" in pre_results:
            cond = pre_results["mlp_bank"]["momentum_cond"]
            print(f"\nmlp_bank pre-step 动量条件数: {cond:.2e} (预期 > 100)")
            assert cond > 100.0, f"mlp_bank 动量应病态，但 cond={cond:.2e}"

        # check post step momentum condition number of mlp_bank is still ill-conditioned
        if "mlp_bank" in pre_results:
            pre_cond = pre_results["mlp_bank"]["momentum_cond"]
            post_cond = post_results["mlp_bank"]["momentum_cond"]
            print(f"mlp_bank 动量条件数变化: {pre_cond:.2e} -> {post_cond:.2e}")

        print("\n" + "=" * 80)
        print("✅ all tests passed！")
        print("=" * 80)

    dist.barrier()
    dist.destroy_process_group()


def run_test():
    world_size = min(2, torch.cuda.device_count())
    print(f"Use {world_size} GPUs for testing")

    mp.spawn(
        test_training_manager_orthogonality,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    run_test()
