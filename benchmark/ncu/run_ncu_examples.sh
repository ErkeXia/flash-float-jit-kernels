#!/usr/bin/env bash
# Profile triton_moun native and TVM-FFI launchers with Nsight Compute.
#
# Run from the repository root on the CUDA VM:
#
#   bash benchmark/ncu/run_ncu_examples.sh
#
# The Python target calls cudaProfilerStart/Stop, so --profile-from-start off
# keeps warmup, Triton compilation, and TVM-FFI inline compilation out of the
# measured range.

set -euo pipefail

M="${M:-4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-1}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-benchmark/ncu/reports}"
NCU="${NCU:-ncu}"
NCU_SUDO="${NCU_SUDO:-auto}"

should_reexec_with_sudo() {
  if [[ "${NCU_SUDO}" == "0" || "${NCU_SUDO}" == "false" ]]; then
    return 1
  fi
  if [[ "${EUID}" -eq 0 || "${FFJK_NCU_SUDO_REEXEC:-0}" == "1" ]]; then
    return 1
  fi
  if [[ "${NCU_SUDO}" == "1" || "${NCU_SUDO}" == "true" ]]; then
    return 0
  fi
  [[ -r /proc/driver/nvidia/params ]] &&
    grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params
}

if should_reexec_with_sudo; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Nsight Compute counters appear to require admin privileges, but sudo was not found." >&2
  else
    CUDA_HOME_VALUE="${CUDA_HOME:-}"
    SUDO_PATH="${PWD}/.venv/bin:${PATH}"
    SUDO_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    if [[ -n "${CUDA_HOME_VALUE}" ]]; then
      SUDO_PATH="${PWD}/.venv/bin:${CUDA_HOME_VALUE}/bin:${PATH}"
      SUDO_LD_LIBRARY_PATH="${CUDA_HOME_VALUE}/lib64:${CUDA_HOME_VALUE}/extras/CUPTI/lib64:${SUDO_LD_LIBRARY_PATH}"
    fi

    echo "Nsight Compute counters require admin privileges; re-running with sudo."
    exec sudo -E env \
      PATH="${SUDO_PATH}" \
      CUDA_HOME="${CUDA_HOME_VALUE}" \
      LD_LIBRARY_PATH="${SUDO_LD_LIBRARY_PATH}" \
      FFJK_NCU_SUDO_REEXEC=1 \
      bash "$0" "$@"
  fi
fi

COMMON_NCU_ARGS=(
  --set detailed
  --target-processes all
  --profile-from-start off
  --launch-count "${LAUNCH_COUNT}"
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

restore_output_ownership() {
  if [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" && -d "${OUT_DIR}" ]]; then
    chown -R "${SUDO_UID}:${SUDO_GID}" "${OUT_DIR}" ||
      echo "Could not restore ownership for ${OUT_DIR}" >&2
  fi
}

run_one triton_moun_native_warm "triton_moun_native_m${M}"
run_one triton_moun_tvm_ffi_warm "triton_moun_tvm_ffi_m${M}"
restore_output_ownership

cat <<EOF

Nsight Compute reports:
  ${OUT_DIR}/triton_moun_native_m${M}.ncu-rep
  ${OUT_DIR}/triton_moun_tvm_ffi_m${M}.ncu-rep

Useful knobs:
  M=2048 WARMUP=5 ITERS=1 bash benchmark/ncu/run_ncu_examples.sh
  NCU_SUDO=1 bash benchmark/ncu/run_ncu_examples.sh
  NCU_SUDO=0 bash benchmark/ncu/run_ncu_examples.sh
EOF
