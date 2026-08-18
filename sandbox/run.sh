#!/bin/bash
# Quick launcher for flash-float-jit-kernels sbx sandbox.
# Usage:
#   ./sandbox/run.sh                          # Launch opencode in sandbox
#   ./sandbox/run.sh --name my-exp            # Named sandbox
#   ./sandbox/run.sh --clone                  # Clone mode (safer)
#   ./sandbox/run.sh shell                    # Shell inside sandbox
#   ./sandbox/run.sh headless                 # Start opencode serve at port 8096
#   ./sandbox/run.sh headless --port 8097     # Custom port
#   ./sandbox/run.sh kernel topk              # Run kernel_agent on TopK via headless
#   ./sandbox/run.sh kernel gemm --max-iter 20
#   ./sandbox/run.sh agent topk               # (legacy) Run kernel agent on TopK
#   ./sandbox/run.sh modal topk               # Run on Modal cloud GPU
#   ./sandbox/run.sh stop                     # Stop sandbox
#   ./sandbox/run.sh rm                       # Remove sandbox
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KIT_PATH="$REPO_ROOT/sbx-kits/flash-kernel-kit"

SBX_NAME="${SBX_NAME:-flash-dev}"
MODE="opencode"
CLONE_FLAG=""
PORT=8096
MAX_ITER=5
AGENT="kernel-dev"
TASK="Optimize this kernel for better performance on Hopper."

# ─── Arg parsing ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)     SBX_NAME="$2"; shift 2 ;;
        --clone)    CLONE_FLAG="--clone"; shift ;;
        --port)     PORT="$2"; shift 2 ;;
        --max-iter) MAX_ITER="$2"; shift 2 ;;
        --agent)    AGENT="$2"; shift 2 ;;
        --task)     TASK="$2"; shift 2 ;;
        stop)       MODE="stop"; shift ;;
        rm)         MODE="remove"; shift ;;
        shell)      MODE="shell"; shift ;;
        headless)   MODE="headless"; shift ;;
        kernel)     MODE="kernel"; shift ;;
        agent)      MODE="agent"; shift ;;
        modal)      MODE="modal"; shift ;;
        --dry)      DRY_RUN="--dry-run"; shift ;;
        topk)       KERNEL_FILE="jit_kernel/csrc/topk_indexer/topk_indexer_radix.cu"; shift ;;
        gemm)       KERNEL_FILE="jit_kernel/csrc/thunder_moun/symm_gemm.cu"; shift ;;
        tritongemm) KERNEL_FILE="jit_kernel/triton3_4/symm_gemm.py"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Default kernel for agent/kernel modes
KERNEL_FILE="${KERNEL_FILE:-jit_kernel/triton3_4/symm_gemm.py}"

echo "=== Flash Float sbx Sandbox ==="

case "$MODE" in
    opencode)
        echo "Launching opencode in sandbox..."
        echo "Kit: $KIT_PATH"
        echo "Name: $SBX_NAME"
        echo ""
        cd "$REPO_ROOT"
        sbx run $CLONE_FLAG opencode --kit "$KIT_PATH" --name "$SBX_NAME"
        ;;

    headless)
        echo "Starting opencode headless server in sandbox..."
        echo "Port: $PORT"
        echo "Name: $SBX_NAME"
        echo ""
        sbx exec "$SBX_NAME" -- bash -c \
            "cd /workspace && opencode serve --port $PORT --hostname 0.0.0.0"
        ;;

    kernel)
        echo "Running kernel agent against headless opencode..."
        echo "Kernel: $KERNEL_FILE"
        echo "Max iterations: $MAX_ITER"
        echo "Agent: $AGENT"
        echo ""
        sbx exec "$SBX_NAME" -- bash -c \
            "cd /workspace && python tools/kernel_agent.py run \
                --kernel '$KERNEL_FILE' \
                --task '$TASK' \
                --max-iter $MAX_ITER \
                --port $PORT \
                --agent $AGENT"
        ;;

    stop)
        echo "Stopping sandbox: $SBX_NAME"
        sbx stop "$SBX_NAME" 2>/dev/null || echo "  (not running)"
        ;;

    remove)
        echo "Removing sandbox: $SBX_NAME"
        sbx rm "$SBX_NAME" 2>/dev/null || echo "  (not found)"
        ;;

    shell)
        echo "Opening shell in sandbox: $SBX_NAME"
        sbx exec -it "$SBX_NAME" bash
        ;;

    agent)
        echo "Running kernel agent in sandbox (legacy mode)..."
        echo "Kernel: $KERNEL_FILE"
        echo "Dry run: ${DRY_RUN:-no}"
        echo ""
        sbx exec "$SBX_NAME" -- python tools/kernel_agent.py harness \
            --kernel "$KERNEL_FILE"
        ;;

    modal)
        echo "Running on Modal cloud GPU..."
        echo "Kernel: $KERNEL_FILE"
        echo ""
        python "$REPO_ROOT/tools/modal_sandbox.py" \
            --kernel-file "$KERNEL_FILE" \
            --gpu h100 \
            --max-iterations 20
        ;;
esac
