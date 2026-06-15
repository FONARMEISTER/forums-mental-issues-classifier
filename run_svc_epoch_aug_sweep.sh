#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="sweep_logs"
mkdir -p "${LOG_DIR}"

for epochs in {1..10}; do
  for use_aug in 0 1; do
    if [[ "${use_aug}" == "1" ]]; then
      aug_name="aug"
    else
      aug_name="noaug"
    fi

    run_name="epochs_${epochs}_${aug_name}"
    out_log="${LOG_DIR}/${run_name}.log"
    err_log="${LOG_DIR}/${run_name}.err"

    echo "============================================================"
    echo "Running ${run_name}"
    echo "stdout: ${out_log}"
    echo "stderr: ${err_log}"
    echo "============================================================"

    python3 svc_filtered_dataset.py \
      --epochs "${epochs}" \
      --use-augmentation "${use_aug}" \
      > "${out_log}" \
      2> "${err_log}"
  done
done

echo "All sweep runs completed. Logs are in ${LOG_DIR}/"
