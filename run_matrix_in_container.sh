#!/usr/bin/env bash
set -euo pipefail
umask 000

experiment_dir="/workspace/gpt-neox/logs/fused_cross_entropy_testing"
config_dir="${experiment_dir}/configs"
run_dir="${experiment_dir}/runs"
source_dir="/workspace/gpt-neox"
parallelism="${PARALLELISM:-tp2}"
run_label="${RUN_LABEL:-}"
vocab_divisibility="${VOCAB_DIVISIBILITY:-128}"
num_layers="${NUM_LAYERS:-16}"
hidden_size="${HIDDEN_SIZE:-2048}"
num_attention_heads="${NUM_ATTENTION_HEADS:-16}"
run_key="${parallelism}${run_label:+_${run_label}}"
train_iters="${TRAIN_ITERS:-50}"
warmup_iters="${WARMUP_ITERS:-10}"
variant_filter="${VARIANT_FILTER:-}"
run_order="${RUN_ORDER:-unfused_first}"
wandb_project="${WANDB_PROJECT:-gpt-neox-fused-cross-entropy-issue-1444}"
wandb_group="${WANDB_GROUP:-1b-wikitext2-${parallelism}-$(date --utc +%Y%m%dT%H%M%SZ)}"
work_dir="$(mktemp -d /tmp/fused-ce-testing-source.XXXXXX)"

case "${parallelism}" in
  tp1|tp2|tp4|tp8) ;;
  *)
    echo "PARALLELISM must be 'tp1', 'tp2', 'tp4', or 'tp8'; got '${parallelism}'." >&2
    exit 2
    ;;
esac

if ! [[ "${vocab_divisibility}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VOCAB_DIVISIBILITY must be a positive integer, got '${vocab_divisibility}'." >&2
  exit 2
fi

for model_dimension in "${num_layers}" "${hidden_size}" "${num_attention_heads}"; do
  if ! [[ "${model_dimension}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_LAYERS, HIDDEN_SIZE, and NUM_ATTENTION_HEADS must be positive integers." >&2
    exit 2
  fi
done

if (( hidden_size % num_attention_heads != 0 )); then
  echo "HIDDEN_SIZE must be divisible by NUM_ATTENTION_HEADS." >&2
  exit 2
fi

if ! [[ "${run_label}" =~ ^[A-Za-z0-9._-]*$ ]]; then
  echo "RUN_LABEL may contain only letters, digits, dots, underscores, and hyphens." >&2
  exit 2
fi

case "${run_order}" in
  unfused_first|fused_first) ;;
  *)
    echo "RUN_ORDER must be 'unfused_first' or 'fused_first', got '${run_order}'." >&2
    exit 2
    ;;
esac

case "${variant_filter}" in
  ""|unfused|fused) ;;
  *)
    echo "VARIANT_FILTER must be empty, 'unfused', or 'fused'; got '${variant_filter}'." >&2
    exit 2
    ;;
esac

if ! [[ "${train_iters}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_ITERS must be a positive integer, got '${train_iters}'." >&2
  exit 2
fi

if ! [[ "${warmup_iters}" =~ ^[0-9]+$ ]] || (( warmup_iters >= train_iters )); then
  echo "WARMUP_ITERS must be a non-negative integer smaller than TRAIN_ITERS." >&2
  exit 2
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY was not passed into the container; refusing an unlogged run." >&2
  exit 2
fi

data_prefix="${source_dir}/logs/inter_document_attention_1b_smoke/data/wikitext2_raw/wikitext2_raw_text_document"
for required_file in \
  "${data_prefix}.bin" \
  "${data_prefix}.idx" \
  "${source_dir}/logs/inter_document_attention_1b_smoke/data/gpt2-vocab.json" \
  "${source_dir}/logs/inter_document_attention_1b_smoke/data/gpt2-merges.txt"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required test data is missing: ${required_file}" >&2
    echo "See ${experiment_dir}/README.md for preparation instructions." >&2
    exit 2
  fi
done

mkdir -p "${run_dir}"
summary_path="${experiment_dir}/summary_${run_key}.tsv"
comparison_path="${experiment_dir}/comparison_${run_key}.tsv"
printf 'variant\texit_code\tresult\tduration_seconds\tfinal_lm_loss\tavg_iteration_ms\tavg_samples_per_sec\n' \
  > "${summary_path}"
printf 'parallelism\tfinal_loss_abs_diff\tunfused_over_fused_speedup\n' \
  > "${comparison_path}"
tar \
  --exclude='./.git' \
  --exclude='./logs' \
  --exclude='./wandb' \
  --exclude='./__pycache__' \
  --exclude='*.pyc' \
  -C "${source_dir}" -cf - . \
  | tar -C "${work_dir}" -xf -
cd "${work_dir}"

unexpected=0

average_iteration_ms() {
  local log_path="$1"
  awk -F '[|]' -v warmup="${warmup_iters}" '
    /elapsed time per iteration \(ms\):/ {
      iteration = -1
      elapsed = -1
      for (field = 1; field <= NF; field++) {
        value = $field
        if (value ~ /^[[:space:]]*iteration[[:space:]]/) {
          sub(/^[[:space:]]*iteration[[:space:]]+/, "", value)
          sub(/\/.*/, "", value)
          gsub(/[[:space:]]/, "", value)
          iteration = value + 0
        } else if (value ~ /elapsed time per iteration \(ms\):/) {
          sub(/^.*:[[:space:]]*/, "", value)
          gsub(/[[:space:]]/, "", value)
          elapsed = value + 0
        }
      }
      if (iteration > warmup && elapsed >= 0) {
        total += elapsed
        count += 1
      }
    }
    END {
      if (count > 0) {
        printf "%.3f", total / count
      } else {
        printf "-"
      }
    }
  ' "${log_path}"
}

average_samples_per_sec() {
  local log_path="$1"
  awk -F '[|]' -v warmup="${warmup_iters}" '
    /elapsed time per iteration \(ms\):/ {
      iteration = -1
      samples = -1
      for (field = 1; field <= NF; field++) {
        value = $field
        if (value ~ /^[[:space:]]*iteration[[:space:]]/) {
          sub(/^[[:space:]]*iteration[[:space:]]+/, "", value)
          sub(/\/.*/, "", value)
          gsub(/[[:space:]]/, "", value)
          iteration = value + 0
        } else if (value ~ /^[[:space:]]*samples\/sec:/) {
          sub(/^.*:[[:space:]]*/, "", value)
          gsub(/[[:space:]]/, "", value)
          samples = value + 0
        }
      }
      if (iteration > warmup && samples >= 0) {
        total += samples
        count += 1
      }
    }
    END {
      if (count > 0) {
        printf "%.3f", total / count
      } else {
        printf "-"
      }
    }
  ' "${log_path}"
}

run_variant() {
  local variant="$1"
  local log_path="${run_dir}/${run_key}_${variant}.log"
  local start_time
  local end_time
  local duration
  local status
  local result
  local final_loss="-"
  local avg_iteration_ms="-"
  local avg_samples_per_sec="-"
  local override_config="${work_dir}/fused_ce_test_overrides_${parallelism}_${variant}.yml"
  local wandb_run_name="${wandb_group}-${variant}"

  if [[ -n "${variant_filter}" && "${variant}" != "${variant_filter}" ]]; then
    return
  fi

  start_time="$(date +%s)"
  echo "[$(date --iso-8601=seconds)] Starting ${parallelism}_${variant}" \
    | tee "${log_path}"
  printf '{"num_layers": %s, "hidden_size": %s, "num_attention_heads": %s, "attention_config": [[["flash"], %s]], "train_iters": %s, "lr_decay_iters": %s, "make_vocab_size_divisible_by": %s, "wandb_project": "%s", "wandb_group": "%s", "wandb_run_name": "%s"}\n' \
    "${num_layers}" "${hidden_size}" "${num_attention_heads}" "${num_layers}" \
    "${train_iters}" "${train_iters}" "${vocab_divisibility}" "${wandb_project}" "${wandb_group}" "${wandb_run_name}" \
    > "${override_config}"
  set +e
  python deepy.py train.py \
    "${config_dir}/base.yml" \
    "${config_dir}/${parallelism}.yml" \
    "${config_dir}/${variant}.yml" \
    "${override_config}" \
    2>&1 | tee -a "${log_path}"
  status="${PIPESTATUS[0]}"
  set -e
  end_time="$(date +%s)"
  duration="$((end_time - start_time))"

  if [[ "${status}" -eq 0 ]] \
    && grep -Eq "iteration[[:space:]]+${train_iters}/[[:space:]]+${train_iters}" "${log_path}" \
    && ! grep -Eq 'number of nan iterations:[[:space:]]+[1-9]' "${log_path}" \
    && grep -Eq 'wandb: Synced |https://wandb.ai/.*/runs/' "${log_path}"; then
    result="PASS"
    final_loss="$(grep -Eo 'lm_loss: [^ |]+' "${log_path}" | tail -n 1 | awk '{print $2}')"
    avg_iteration_ms="$(average_iteration_ms "${log_path}")"
    avg_samples_per_sec="$(average_samples_per_sec "${log_path}")"
  else
    result="FAIL"
    unexpected=$((unexpected + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_key}_${variant}" \
    "${status}" \
    "${result}" \
    "${duration}" \
    "${final_loss}" \
    "${avg_iteration_ms}" \
    "${avg_samples_per_sec}" \
    >> "${summary_path}"
  chmod a+rw "${log_path}" "${summary_path}" 2>/dev/null || true
  echo "[$(date --iso-8601=seconds)] Finished ${parallelism}_${variant}: ${result} (${duration}s)" \
    | tee -a "${log_path}"
}

if [[ "${run_order}" == "unfused_first" ]]; then
  run_variant unfused
  run_variant fused
else
  run_variant fused
  run_variant unfused
fi

unfused_row="$(awk -F '\t' -v variant="${run_key}_unfused" '$1 == variant && $3 == "PASS" {print; exit}' "${summary_path}")"
fused_row="$(awk -F '\t' -v variant="${run_key}_fused" '$1 == variant && $3 == "PASS" {print; exit}' "${summary_path}")"
if [[ -n "${unfused_row}" && -n "${fused_row}" ]]; then
  unfused_loss="$(awk -F '\t' '{print $5}' <<< "${unfused_row}")"
  fused_loss="$(awk -F '\t' '{print $5}' <<< "${fused_row}")"
  unfused_ms="$(awk -F '\t' '{print $6}' <<< "${unfused_row}")"
  fused_ms="$(awk -F '\t' '{print $6}' <<< "${fused_row}")"
  loss_abs_diff="$(awk -v lhs="${unfused_loss}" -v rhs="${fused_loss}" 'BEGIN {diff = lhs - rhs; if (diff < 0) diff = -diff; printf "%.9g", diff}')"
  speedup="$(awk -v baseline="${unfused_ms}" -v fused="${fused_ms}" 'BEGIN {if (fused > 0) printf "%.6f", baseline / fused; else printf "-"}')"
  printf '%s\t%s\t%s\n' "${run_key}" "${loss_abs_diff}" "${speedup}" \
    >> "${comparison_path}"
fi
chmod a+rw "${comparison_path}" 2>/dev/null || true

echo
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${summary_path}"
  if [[ -n "${unfused_row}" && -n "${fused_row}" ]]; then
    echo
    column -t -s $'\t' "${comparison_path}"
  fi
else
  sed -n '1,200p' "${summary_path}"
  if [[ -n "${unfused_row}" && -n "${fused_row}" ]]; then
    echo
    sed -n '1,200p' "${comparison_path}"
  fi
fi

if [[ "${unexpected}" -ne 0 ]]; then
  echo "${unexpected} fused cross-entropy training variant(s) failed." >&2
  exit 1
fi

echo "All selected ${parallelism} fused cross-entropy variants passed."
