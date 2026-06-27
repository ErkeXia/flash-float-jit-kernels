"""Minimal Nsight Systems target for symmetric GEMM providers.

This script builds the same inputs as the profiling benchmarks, warms up one
provider, then runs a small fixed number of launches under an optional CUDA
profiler API capture range. It is intentionally host-side and non-invasive:
the kernels are not modified.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PROFILING_DIR = REPO_ROOT / "benchmark" / "profiling"
for path in (REPO_ROOT, PROFILING_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from workload import (  # noqa: E402
    DEFAULT_SCENARIOS,
    make_provider,
    prepare_inputs,
    scenario_config,
    triton_ffi_mode,
    warmup,
)


def cuda_profiler_start(enabled: bool) -> None:
    if not enabled:
        return
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime.
        print(f"warning: cudaProfilerStart failed: {exc}", file=sys.stderr)


def cuda_profiler_stop(enabled: bool) -> None:
    if not enabled:
        return
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime.
        print(f"warning: cudaProfilerStop failed: {exc}", file=sys.stderr)


def nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one symmetric GEMM provider for Nsight Systems timeline "
            "profiling. Use nsys --capture-range=cudaProfilerApi to exclude "
            "warmup and Triton compilation from the trace."
        )
    )
    parser.add_argument(
        "--scenario",
        default="cuda_warm",
        choices=DEFAULT_SCENARIOS,
        help="Scenario/provider to launch.",
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cold-iters", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--poison-output",
        action="store_true",
        help="Fill output before the profiled launches and print remaining sentinels.",
    )
    parser.add_argument("--poison-value", type=float, default=12345.0)
    parser.add_argument(
        "--no-profiler-api",
        action="store_false",
        dest="profiler_api",
        help=(
            "Do not call cudaProfilerStart/Stop. Use this only if the nsys "
            "command captures from process start."
        ),
    )
    parser.set_defaults(profiler_api=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight Systems profiling.")
    if args.iters < 1:
        raise ValueError("--iters must be at least 1")

    tensors = prepare_inputs(args.m, args.device)
    config = scenario_config(args.scenario, args)
    out = torch.empty((args.m, args.m), device=args.device, dtype=torch.float16)
    fn = make_provider(str(config["provider"]), tensors, out)
    use_tvm_ffi = config["triton_use_tvm_ffi"]

    print(
        "nsys target:",
        f"scenario={args.scenario}",
        f"provider={config['provider']}",
        f"launcher={config['launcher']}",
        f"m={args.m}",
        f"warmup={config['warmup']}",
        f"iters={args.iters}",
        f"triton_use_tvm_ffi={use_tvm_ffi}",
        flush=True,
    )

    with triton_ffi_mode(use_tvm_ffi if isinstance(use_tvm_ffi, bool) else None):
        if int(config["warmup"]) > 0:
            nvtx_push(f"{args.scenario}/warmup")
            try:
                warmup(fn, int(config["warmup"]))
            finally:
                nvtx_pop()

        if args.poison_output:
            out.fill_(args.poison_value)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        cuda_profiler_start(args.profiler_api)
        nvtx_push(f"{args.scenario}/profile")
        wall_start = time.perf_counter()
        try:
            for i in range(args.iters):
                nvtx_push(f"{args.scenario}/iter_{i}")
                try:
                    fn()
                finally:
                    nvtx_pop()
            torch.cuda.synchronize()
        finally:
            elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
            nvtx_pop()
            cuda_profiler_stop(args.profiler_api)

    print(f"wall_ms_total={elapsed_ms:.6f}", flush=True)
    print(f"wall_ms_per_iter={elapsed_ms / args.iters:.6f}", flush=True)

    if args.poison_output:
        remaining = (out == args.poison_value).sum().item()
        print(f"sentinel_remaining={remaining}", flush=True)


if __name__ == "__main__":
    main()
