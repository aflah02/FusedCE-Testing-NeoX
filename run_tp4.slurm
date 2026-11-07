#!/bin/bash -l

# Submit from the GPT-NeoX repository root. Override the default partition with
# `sbatch --partition=h200 ...` (also valid: h100, h100nvl, h100sxm).
#SBATCH --job-name=fused-ce-tp4
#SBATCH --partition=h100,h100nvl,h100sxm,h200
#SBATCH --exclude=sws-8h100grid-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${GPT_NEOX_REPO:-}" ]]; then
  repo_dir="${GPT_NEOX_REPO}"
elif [[ -f "${submit_dir}/logs/fused_cross_entropy_testing/run_matrix.sh" ]]; then
  repo_dir="${submit_dir}"
elif [[ -f "${submit_dir}/run_matrix.sh" ]] \
  && [[ "$(basename -- "${submit_dir}")" == "fused_cross_entropy_testing" ]]; then
  repo_dir="$(cd -- "${submit_dir}/../.." && pwd)"
else
  echo "Could not locate the GPT-NeoX repository from ${submit_dir}." >&2
  echo "Submit from the repository root or logs/fused_cross_entropy_testing," >&2
  echo "or export GPT_NEOX_REPO=/absolute/repo/path." >&2
  exit 2
fi

experiment_dir="${repo_dir}/logs/fused_cross_entropy_testing"
launcher="${experiment_dir}/run_matrix.sh"
image_name="${GPT_NEOX_IMAGE:-gpt-neox:te}"
image_tar="${GPT_NEOX_IMAGE_TAR:-/NS/llm-pretraining/work/afkhan/GPT-NeoX-Dev/TorchRun_Support/gpt-neox-te.tar}"
allocated_gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ ! -f "${launcher}" ]]; then
  echo "Could not find ${launcher}." >&2
  echo "Export GPT_NEOX_REPO=/absolute/repo/path." >&2
  exit 2
fi

IFS=',' read -r -a allocated_gpu_list <<< "${allocated_gpu_devices}"
if [[ "${#allocated_gpu_list[@]}" -ne 4 ]]; then
  echo "Expected four allocated GPUs; CUDA_VISIBLE_DEVICES=${allocated_gpu_devices}." >&2
  exit 2
fi

cd "${repo_dir}"

echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Node: $(hostname)"
echo "Repository: ${repo_dir}"
echo "Allocated GPU devices: ${allocated_gpu_devices}"
echo "Start: $(date --iso-8601=seconds)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable on the allocated node." >&2
  exit 2
fi

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
nvidia-smi topo -m

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is unavailable on the allocated node." >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "The Docker daemon is unavailable for $(id -un) on $(hostname)." >&2
  exit 2
fi

if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
  if [[ ! -r "${image_tar}" ]]; then
    echo "Docker image ${image_name} is absent and ${image_tar} is unreadable." >&2
    exit 2
  fi
  echo "Loading ${image_name} from ${image_tar}; no image will be built."
  docker load --input "${image_tar}"
fi

export PARALLELISM=tp4
export GPU_DEVICES="${allocated_gpu_devices}"
base_run_label="${RUN_LABEL:-${SLURM_JOB_PARTITION:-tp4}-${SLURM_JOB_ID:-manual}}"
vocab_sizes_spec="${VOCAB_SIZES:-50432:65536:131072:262144}"
export TRAIN_ITERS="${TRAIN_ITERS:-50}"
export WARMUP_ITERS="${WARMUP_ITERS:-10}"
export RUN_ORDER="${RUN_ORDER:-unfused_first}"
export NUM_LAYERS="${NUM_LAYERS:-16}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
export WANDB_PROJECT="${WANDB_PROJECT:-gpt-neox-fused-cross-entropy-issue-1444}"
export GPT_NEOX_IMAGE="${image_name}"

IFS=':' read -r -a vocab_sizes <<< "${vocab_sizes_spec}"
if [[ "${#vocab_sizes[@]}" -eq 0 ]]; then
  echo "VOCAB_SIZES must contain at least one supported padded vocabulary size." >&2
  exit 2
fi

echo "Base run label: ${base_run_label}"
echo "Iterations: ${TRAIN_ITERS} (${WARMUP_ITERS} warmup)"
echo "Model: layers=${NUM_LAYERS}, hidden=${HIDDEN_SIZE}, heads=${NUM_ATTENTION_HEADS}"
echo "Padded vocabulary sweep: ${vocab_sizes_spec}"

for vocab_size in "${vocab_sizes[@]}"; do
  case "${vocab_size}" in
    50432) vocab_divisibility=12608 ;;
    65536) vocab_divisibility=16384 ;;
    131072) vocab_divisibility=32768 ;;
    262144) vocab_divisibility=65536 ;;
    *)
      echo "Unsupported TP=4 padded vocabulary size: ${vocab_size}." >&2
      echo "Supported sizes: 50432,65536,131072,262144." >&2
      exit 2
      ;;
  esac

  # NeoX pads to make_vocab_size_divisible_by * tensor_parallel_size.
  # These TP=4 divisibility values reproduce the requested padded sizes.
  export RUN_LABEL="${base_run_label}-vocab${vocab_size}"
  export VOCAB_DIVISIBILITY="${vocab_divisibility}"

  echo
  echo "Starting padded vocab ${vocab_size} with divisibility ${VOCAB_DIVISIBILITY}."
  bash "${launcher}"
done

echo "Finish: $(date --iso-8601=seconds)"
