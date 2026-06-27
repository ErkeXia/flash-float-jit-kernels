#!/usr/bin/env bash
# Example Nsight Compute commands for the symmetric GEMM providers.
#
# Run from the repository root on the CUDA VM:
#
#   bash benchmark/ncu/run_ncu_examples.sh
#
# The target script calls cudaProfilerStart/Stop, so these commands use
# --profile-from-start off to keep warmup and Triton compilation out of the
# profiled region.

set -euo pipefail

M="${M:-4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-benchmark/ncu/reports}"
NCU="${NCU:-ncu}"

COMMON_NCU_ARGS=(
  --set detailed
  --target-processes all
  --profile-from-start off
  --launch-count 1
  --kernel-name-base demangled
  --force-overwrite
)

run_one() {
  local scenario="$1"
  local output_name="$2"
  mkdir -p "${OUT_DIR}"
  "${NCU}" "${COMMON_NCU_ARGS[@]}" \
    -o "${OUT_DIR}/${output_name}" \
    "${PYTHON}" benchmark/ncu/ncu_target.py \
      --scenario "${scenario}" \
      --m "${M}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --poison-output
}

run_one cuda_warm "cuda_m${M}"
run_one triton_symm_native_warm "triton_symm_m${M}"

cat <<EOF

Reports:
  ${OUT_DIR}/cuda_m${M}.ncu-rep
  ${OUT_DIR}/triton_symm_m${M}.ncu-rep

Optional raw CSV export:
  ${NCU} -i ${OUT_DIR}/cuda_m${M}.ncu-rep --page raw --csv > ${OUT_DIR}/cuda_m${M}_raw.csv
  ${NCU} -i ${OUT_DIR}/cuda_m${M}.ncu-rep --page source --csv > ${OUT_DIR}/cuda_m${M}_source.csv
  ${NCU} -i ${OUT_DIR}/triton_symm_m${M}.ncu-rep --page raw --csv > ${OUT_DIR}/triton_symm_m${M}_raw.csv
  ${NCU} -i ${OUT_DIR}/triton_symm_m${M}.ncu-rep --page source --csv > ${OUT_DIR}/triton_symm_m${M}_source.csv

If NVIDIA performance counters require admin privileges, run with sudo while
preserving the virtualenv and CUDA paths, for example:

  sudo -E env \\
    PATH="\$PWD/.venv/bin:\$CUDA_HOME/bin:\$PATH" \\
    CUDA_HOME="\$CUDA_HOME" \\
    LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/extras/CUPTI/lib64:\$LD_LIBRARY_PATH" \\
    bash benchmark/ncu/run_ncu_examples.sh
EOF
