#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${experiment_dir}/../.." && pwd)"
image_name="${GPT_NEOX_IMAGE:-gpt-neox:te}"
parallelism="${PARALLELISM:-tp2}"
run_label="${RUN_LABEL:-}"
vocab_divisibility="${VOCAB_DIVISIBILITY:-128}"
num_layers="${NUM_LAYERS:-16}"
hidden_size="${HIDDEN_SIZE:-2048}"
num_attention_heads="${NUM_ATTENTION_HEADS:-16}"
train_iters="${TRAIN_ITERS:-50}"
warmup_iters="${WARMUP_ITERS:-10}"
variant_filter="${VARIANT_FILTER:-}"
run_order="${RUN_ORDER:-unfused_first}"
host_uid="${HOST_UID:-$(id -u)}"
host_gid="${HOST_GID:-$(id -g)}"
container_user="${CONTAINER_USER:-}"
wandb_project="${WANDB_PROJECT:-gpt-neox-fused-cross-entropy-issue-1444}"
run_descriptor="${parallelism}${run_label:+-${run_label}}"
wandb_group="${WANDB_GROUP:-1b-wikitext2-${run_descriptor}-$(date --utc +%Y%m%dT%H%M%SZ)}"
wandb_runtime_version="${WANDB_RUNTIME_VERSION:-0.22.3}"
wandb_netrc_path="${WANDB_NETRC_PATH:-${HOME}/.netrc}"
container_home="/tmp/fused-ce-testing-home"
wandb_auth_args=()

if [[ -z "${container_user}" ]]; then
  if docker info --format '{{json .SecurityOptions}}' | grep -q 'name=rootless'; then
    # Container root maps to the invoking host user under rootless Docker. This
    # is required for writes to NFS mounts, which reject subordinate UIDs.
    container_user="0:0"
  else
    container_user="${host_uid}:${host_gid}"
  fi
fi

case "${parallelism}" in
  tp1)
    gpu_devices="${GPU_DEVICES:-${GPU_DEVICE:-0}}"
    expected_gpu_count=1
    ;;
  tp2)
    gpu_devices="${GPU_DEVICES:-0,1}"
    expected_gpu_count=2
    ;;
  tp4)
    gpu_devices="${GPU_DEVICES:-0,1,2,3}"
    expected_gpu_count=4
    ;;
  tp8)
    gpu_devices="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
    expected_gpu_count=8
    ;;
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

IFS=',' read -r -a fused_ce_gpu_list <<< "${gpu_devices}"
if [[ "${#fused_ce_gpu_list[@]}" -ne "${expected_gpu_count}" ]]; then
  echo "${parallelism} requires ${expected_gpu_count} GPU device(s); got '${gpu_devices}'." >&2
  exit 2
fi

if [[ "${gpu_devices}" == *,* ]]; then
  gpu_request="\"device=${gpu_devices}\""
else
  gpu_request="device=${gpu_devices}"
fi

echo "Verifying ${expected_gpu_count} benchmark GPU(s): ${gpu_devices}"
docker run --rm \
  --gpus "${gpu_request}" \
  "${image_name}" \
  python -c \
  "import torch; count = torch.cuda.device_count(); print('visible benchmark GPUs', count); assert count == ${expected_gpu_count}, f'expected ${expected_gpu_count} GPUs, found {count}'"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb_auth_args+=(--env WANDB_API_KEY)
elif [[ -r "${wandb_netrc_path}" ]]; then
  wandb_api_key="$({
    awk -v target="api.wandb.ai" '
      $1 == "machine" { active = ($2 == target) }
      active {
        for (field = 1; field <= NF; field++) {
          if ($field == "password" && field < NF) {
            print $(field + 1)
            exit
          }
        }
      }
    ' "${wandb_netrc_path}"
  })"
  if [[ -z "${wandb_api_key}" ]]; then
    echo "No api.wandb.ai password found in ${wandb_netrc_path}." >&2
    exit 2
  fi
  export WANDB_API_KEY="${wandb_api_key}"
  wandb_auth_args+=(--env WANDB_API_KEY)
else
  echo "Set WANDB_API_KEY or WANDB_NETRC_PATH to readable W&B credentials." >&2
  exit 2
fi

docker run --rm \
  --ipc=host \
  --user "${container_user}" \
  --gpus "${gpu_request}" \
  --volume "${repo_dir}:/workspace/gpt-neox" \
  --workdir /workspace/gpt-neox \
  --env XDG_CACHE_HOME=/tmp/fused-ce-testing-cache \
  --env HOME="${container_home}" \
  --env TRITON_CACHE_DIR=/tmp/fused-ce-testing-triton \
  --env TORCH_EXTENSIONS_DIR=/tmp/fused-ce-testing-extensions \
  --env HF_HOME=/tmp/fused-ce-testing-hf \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PARALLELISM=${parallelism}" \
  --env "RUN_LABEL=${run_label}" \
  --env "VOCAB_DIVISIBILITY=${vocab_divisibility}" \
  --env "NUM_LAYERS=${num_layers}" \
  --env "HIDDEN_SIZE=${hidden_size}" \
  --env "NUM_ATTENTION_HEADS=${num_attention_heads}" \
  --env "TRAIN_ITERS=${train_iters}" \
  --env "WARMUP_ITERS=${warmup_iters}" \
  --env "VARIANT_FILTER=${variant_filter}" \
  --env "RUN_ORDER=${run_order}" \
  --env "WANDB_PROJECT=${wandb_project}" \
  --env "WANDB_GROUP=${wandb_group}" \
  --env "WANDB_RUNTIME_VERSION=${wandb_runtime_version}" \
  "${wandb_auth_args[@]}" \
  "${image_name}" \
  bash -lc \
  'python -m pip install --quiet --disable-pip-version-check --upgrade "wandb==${WANDB_RUNTIME_VERSION}" typing_extensions && exec bash logs/fused_cross_entropy_testing/run_matrix_in_container.sh'
