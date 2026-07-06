#!/usr/bin/env bash
# Profile triton_moun native and TVM-FFI launchers with Nsight Systems.
#
# Run from the repository root on the CUDA VM:
#
#   bash benchmark/nsys/run_nsys_examples.sh
#
# The Python target calls cudaProfilerStart/Stop, so capture-range keeps warmup,
# Triton compilation, and TVM-FFI inline compilation out of the trace.

set -euo pipefail

M="${M:-4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-10}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-benchmark/nsys/reports}"
NSYS="${NSYS:-nsys}"

COMMON_NSYS_ARGS=(
  profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --force-overwrite=true
)

run_one() {
  local scenario="$1"
  local output_name="$2"
  mkdir -p "${OUT_DIR}"
  "${NSYS}" "${COMMON_NSYS_ARGS[@]}" \
    -o "${OUT_DIR}/${output_name}" \
    "${PYTHON}" benchmark/nsys/nsys_target.py \
      --scenario "${scenario}" \
      --m "${M}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --poison-output
}

run_one triton_moun_native_warm "triton_moun_native_m${M}"
run_one triton_moun_tvm_ffi_warm "triton_moun_tvm_ffi_m${M}"

cat <<EOF

Nsight Systems reports:
  ${OUT_DIR}/triton_moun_native_m${M}.nsys-rep
  ${OUT_DIR}/triton_moun_tvm_ffi_m${M}.nsys-rep

Useful knobs:
  M=2048 WARMUP=5 ITERS=10 bash benchmark/nsys/run_nsys_examples.sh
EOF
