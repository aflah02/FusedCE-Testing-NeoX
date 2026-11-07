#!/bin/bash -l

# Build the Hopper image on the allocated node, then run the TP=4 vocabulary
# sweep. Docker build output and periodic heartbeats are written to both the
# Slurm output and docker-build-${SLURM_JOB_ID}.log.
#SBATCH --job-name=fused-ce-tp4-build
#SBATCH --partition=h100,h100nvl,h100sxm,h200
#SBATCH --exclude=sws-8h100grid-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${GPT_NEOX_REPO:-}" ]]; then
  repo_dir="${GPT_NEOX_REPO}"
elif [[ -f "${submit_dir}/logs/fused_cross_entropy_testing/run_tp4.slurm" ]]; then
  repo_dir="${submit_dir}"
elif [[ -f "${submit_dir}/run_tp4.slurm" ]] \
  && [[ "$(basename -- "${submit_dir}")" == "fused_cross_entropy_testing" ]]; then
  repo_dir="$(cd -- "${submit_dir}/../.." && pwd)"
else
  echo "Could not locate the GPT-NeoX repository from ${submit_dir}." >&2
  echo "Submit from the repository root or logs/fused_cross_entropy_testing," >&2
  echo "or export GPT_NEOX_REPO=/absolute/repo/path." >&2
  exit 2
fi

experiment_dir="${repo_dir}/logs/fused_cross_entropy_testing"
benchmark_script="${experiment_dir}/run_tp4.slurm"
dockerfile="${repo_dir}/containers/docker/Dockerfile.TE"
image_name="${GPT_NEOX_IMAGE:-gpt-neox:te}"
build_log="${experiment_dir}/docker-build-${SLURM_JOB_ID:-manual}.log"
heartbeat_seconds="${BUILD_HEARTBEAT_SECONDS:-60}"
allocated_gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ ! -f "${benchmark_script}" || ! -f "${dockerfile}" ]]; then
  echo "Missing ${benchmark_script} or ${dockerfile}." >&2
  exit 2
fi

if ! [[ "${heartbeat_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUILD_HEARTBEAT_SECONDS must be a positive integer." >&2
  exit 2
fi

IFS=',' read -r -a allocated_gpu_list <<< "${allocated_gpu_devices}"
if [[ "${#allocated_gpu_list[@]}" -ne 4 ]]; then
  echo "Expected four allocated GPUs; CUDA_VISIBLE_DEVICES=${allocated_gpu_devices}." >&2
  exit 2
fi

if [[ "${allocated_gpu_devices}" == *,* ]]; then
  gpu_request="\"device=${allocated_gpu_devices}\""
else
  gpu_request="device=${allocated_gpu_devices}"
fi

cd "${repo_dir}"
exec > >(tee -a "${build_log}") 2>&1

log_step() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

log_step "Starting job ${SLURM_JOB_ID:-manual} on $(hostname)."
log_step "Partition: ${SLURM_JOB_PARTITION:-unknown}"
log_step "Repository: ${repo_dir}"
log_step "Dockerfile: ${dockerfile}"
log_step "Image tag: ${image_name}"
log_step "Allocated GPU devices: ${allocated_gpu_devices}"
log_step "Combined build log: ${build_log}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log_step "ERROR: nvidia-smi is unavailable."
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

if ! command -v docker >/dev/null 2>&1; then
  log_step "ERROR: docker is unavailable."
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  log_step "ERROR: Docker daemon is unavailable for $(id -un) on $(hostname)."
  exit 2
fi

log_step "Docker environment before build:"
docker version
docker info --format 'Docker root={{.DockerRootDir}} driver={{.Driver}}'
docker system df

build_args=(
  --file "${dockerfile}"
  --tag "${image_name}"
)
if [[ "${DOCKER_BUILD_NO_CACHE:-0}" == "1" ]]; then
  build_args+=(--no-cache)
  log_step "Docker layer cache is disabled."
else
  log_step "Docker layer cache is enabled."
fi

export DOCKER_BUILDKIT=0
build_start_seconds="$(date +%s)"
log_step "Starting docker build with the classic builder (DOCKER_BUILDKIT=0)."
log_step "Classic Step N/M output follows."

set +e
docker build "${build_args[@]}" "${repo_dir}" &
build_pid=$!
(
  while kill -0 "${build_pid}" 2>/dev/null; do
    sleep "${heartbeat_seconds}"
    if kill -0 "${build_pid}" 2>/dev/null; then
      log_step "Docker build is still running (pid=${build_pid})."
    fi
  done
) &
heartbeat_pid=$!

wait "${build_pid}"
build_status=$?
kill "${heartbeat_pid}" 2>/dev/null || true
wait "${heartbeat_pid}" 2>/dev/null || true
set -e

build_end_seconds="$(date +%s)"
build_duration="$((build_end_seconds - build_start_seconds))"
if [[ "${build_status}" -ne 0 ]]; then
  log_step "ERROR: docker build failed with status ${build_status} after ${build_duration}s."
  exit "${build_status}"
fi

log_step "Docker build completed in ${build_duration}s."
docker image inspect "${image_name}" \
  --format 'Image={{.Id}} size={{.Size}} bytes created={{.Created}}'
docker system df

log_step "Verifying the built image and GPU passthrough."
docker run --rm --gpus "${gpu_request}" --ipc=host "${image_name}" \
  python -c 'import torch, flash_attn, transformer_engine; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())'

export GPT_NEOX_IMAGE="${image_name}"
log_step "Image verification passed; starting the TP=4 benchmark sweep."
bash "${benchmark_script}"
log_step "TP=4 benchmark sweep finished."
