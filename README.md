# Fused vocabulary-parallel cross-entropy matrix

This directory contains matched end-to-end training runs for NeoX's existing
vocabulary-parallel cross entropy and the optional fused backend added for
issue #1444. The two variants differ only in `cross_entropy_fusion`:

| Variant | `cross_entropy_fusion` | Tensor-parallel CE collectives |
| --- | --- | --- |
| `unfused` | `false` | three (MAX plus two SUM reductions) |
| `fused` | `true` | two (MAX plus one combined SUM reduction) |

The default profile is a 1B-class BF16 model with an approximately 50.3K-token
padded GPT-2 vocabulary, sequence length 2,048, native FlashAttention,
activation checkpointing, no dropout, and no evaluation or checkpoint I/O.
Runs are logged to the dedicated
`gpt-neox-fused-cross-entropy-issue-1444` W&B project. Keeping the output
vocabulary large makes the loss path a measurable part of the step. Both
variants use the same seed and data configuration.

Run both TP=1 and tensor-parallel cases when validating a change. TP=1 is the
control case: it checks the fused math without offering collective-latency
savings. TP=2, TP=4, and TP=8 exercise vocabulary partitioning and the
reduction from three collectives to two.

## Prerequisites

The launchers follow the other experiments under `logs/` and expect:

- a CUDA-capable `gpt-neox:te` Docker image (override with
  `GPT_NEOX_IMAGE`); and
- the prepared WikiText-2 mmap data and GPT-2 tokenizer files under
  `logs/inter_document_attention_1b_smoke/data/`.
- W&B credentials supplied through `WANDB_API_KEY`, or an `api.wandb.ai`
  password in `~/.netrc` (override its location with `WANDB_NETRC_PATH`).

See `logs/inter_document_attention_1b_smoke/README.md` for data preparation.
Evaluation and checkpoint saving are disabled so they do not add periodic I/O.
Override the project with `WANDB_PROJECT`; each launcher invocation creates a
timestamped group by default, which can be overridden with `WANDB_GROUP`.
The saved image contains an obsolete W&B client, so the launcher installs W&B
0.22.3 in the disposable container at runtime. This does not modify or build
the image. Set `WANDB_RUNTIME_VERSION` to test another client version.

## Running

Run the TP=2 comparison on two GPUs:

```bash
PARALLELISM=tp2 GPU_DEVICES=0,1 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh
```

Run the TP=1 control on one GPU:

```bash
PARALLELISM=tp1 GPU_DEVICE=0 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh
```

The same launcher has TP=4 and TP=8 configurations for a larger machine:

```bash
PARALLELISM=tp4 GPU_DEVICES=0,1,2,3 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh

PARALLELISM=tp8 GPU_DEVICES=0,1,2,3,4,5,6,7 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh
```

For a single-node eight-GPU Slurm allocation, submit from the experiment
directory so Slurm writes scheduler output into this repository's `logs/`
directory:

```bash
cd logs/fused_cross_entropy_testing

# Short TP=8 smoke test on H100 SXM; both variants run.
sbatch --partition=h100sxm \
  --export=ALL,TRAIN_ITERS=4,WARMUP_ITERS=1,RUN_LABEL=smoke \
  run_tp8.slurm

# Full TP=8 comparison on H200.
sbatch --partition=h200 \
  --export=ALL,TRAIN_ITERS=50,WARMUP_ITERS=10,RUN_LABEL=h200-full \
  run_tp8.slurm
```

The partition can also be `h100` or `h100nvl`. The job requests one node and
all eight GPUs with a single launcher task. It checks GPU visibility and the
Docker daemon, then loads `gpt-neox:te` from the saved tar only when the image
is absent; it never builds an image. Set `GPT_NEOX_REPO`, `GPT_NEOX_IMAGE`, or
`GPT_NEOX_IMAGE_TAR` through `sbatch --export` to override their defaults.
Both Slurm variants exclude `sws-8h100grid-04`, where image transfer stalled.

By default, one Slurm job runs both unfused and fused training at each of the
four padded vocabulary sizes used by this experiment: 50,432, 65,536,
131,072, and 262,144. At TP=8 their corresponding
`make_vocab_size_divisible_by` values are 6,304, 8,192, 16,384, and 32,768.
Use `VOCAB_SIZES` to run a supported subset, for example:

```bash
sbatch --partition=h200 \
  --export=ALL,VOCAB_SIZES=65536:262144,RUN_LABEL=h200-subset \
  run_tp8.slurm
```

If loading the saved tar is too slow or provides too little progress output,
use the build variant instead:

```bash
sbatch --partition=h200 \
  --export=ALL,TRAIN_ITERS=50,WARMUP_ITERS=10,RUN_LABEL=h200-build \
  run_tp8_build.slurm
```

This builds `gpt-neox:te` from `containers/docker/Dockerfile.TE` on the
allocated node and then runs the same TP=8 vocabulary sweep. BuildKit is
explicitly disabled because the cluster Docker client does not include the
`buildx` component. The classic builder prints each `Step N/M`, and the
wrapper emits a timestamped heartbeat every 60 seconds. Combined output is saved as
`logs/docker-build-<job-id>.log` in addition to
the Slurm output. Existing Docker layers are reused by default; pass
`DOCKER_BUILD_NO_CACHE=1` through `sbatch --export` to force a clean build.

For the corresponding four-GPU sweep, use the TP=4 launchers:

```bash
# Use an existing image, loading the saved tar only when needed.
sbatch --partition=h100 \
  --export=ALL,TRAIN_ITERS=50,WARMUP_ITERS=10,RUN_LABEL=h100-tp4 \
  run_tp4.slurm

# Build the image with the classic builder, then run the TP=4 sweep.
sbatch --partition=h100 \
  --export=ALL,TRAIN_ITERS=50,WARMUP_ITERS=10,RUN_LABEL=h100-tp4-build \
  run_tp4_build.slurm
```

Both TP=4 launchers request four GPUs and 256GB of host memory, exclude
`sws-8h100grid-04`, and sweep the same padded vocabulary sizes as TP=8. The
TP=4 `make_vocab_size_divisible_by` values are 12,608, 16,384, 32,768, and
65,536 for padded vocabularies 50,432, 65,536, 131,072, and 262,144.

The default run uses 50 iterations and excludes the first 10 from timing
averages. Both can be overridden:

```bash
TRAIN_ITERS=100 WARMUP_ITERS=20 PARALLELISM=tp2 GPU_DEVICES=0,1 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh
```

Run only one backend with `VARIANT_FILTER=unfused` or
`VARIANT_FILTER=fused`. By default the unfused case runs first; set
`RUN_ORDER=fused_first` to reverse the order when checking for order-dependent
thermal or cache effects.

Model dimensions can be changed without editing the base config. For example,
the completed 2.018B-parameter comparison used:

```bash
PARALLELISM=tp2 GPU_DEVICES=0,1 RUN_LABEL=model2b \
  NUM_LAYERS=24 HIDDEN_SIZE=2560 NUM_ATTENTION_HEADS=20 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh
```

`VOCAB_DIVISIBILITY` is passed to NeoX as
`make_vocab_size_divisible_by`; it is not a requested vocabulary size. NeoX
rounds the tokenizer vocabulary up to a multiple of
`VOCAB_DIVISIBILITY * TP`. `RUN_LABEL` only names the output files and W&B
group. For example, at TP=2, the following pads the 50,257-token GPT-2
vocabulary to 131,072:

```bash
PARALLELISM=tp2 GPU_DEVICES=0,1 RUN_LABEL=vocab131072 \
  VOCAB_DIVISIBILITY=65536 \
  bash logs/fused_cross_entropy_testing/run_matrix.sh
```

To run inside an existing container whose checkout is mounted at
`/workspace/gpt-neox`, use:

```bash
PARALLELISM=tp2 TRAIN_ITERS=50 WARMUP_ITERS=10 \
  bash logs/fused_cross_entropy_testing/run_matrix_in_container.sh
```

## Outputs

Each backend writes `logs/runs/<tp>_<variant>.log`.
`logs/summary_<tp>.tsv` records the exit code, pass/fail result, wall-clock
duration, final language-model loss, and post-warmup mean iteration time and
samples/second. A run passes only if it reaches the requested final iteration
without a reported NaN iteration.

When both backends run successfully, `logs/comparison_<tp>.tsv` reports:

- the absolute difference between final logged losses; and
- `unfused_ms / fused_ms`, where a value greater than one favors fusion.

The first iterations include model and JIT warmup, which is why they are
excluded from the averages. For performance conclusions, repeat the matrix in
both orders and use a longer run. Matching final losses are a useful
end-to-end signal, while `tests/model/test_fused_cross_entropy.py` remains the
stricter per-token loss/gradient correctness check.

## Results from `sws-2h100-01`

These runs were completed on 2026-10-17 using two NVIDIA H100 PCIe 80GB GPUs,
BF16, sequence length 2,048, micro-batch size 4, activation checkpointing, and
the `gpt-neox:te` image (PyTorch 2.3.0a0, CUDA 12.3). Each timing is the mean
of iterations 11 through 50. The unfused variant ran first in every pair.

All 12 completed jobs reached iteration 50 with zero skipped and zero NaN
iterations and synced to the dedicated [W&B
project](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444).
The four focused tests in `tests/model/test_fused_cross_entropy.py` also
passed in the same image.
The final-loss difference is an end-to-end training sanity check, not the
strict equivalence tolerance used by the unit tests.

| TP | Model parameters | Padded vocab | Unfused ms | Fused ms | Fused speed change | Final-loss absolute difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.909B | 50,304 | 341.697 | 342.552 | -0.25% | 0.002841 |
| 2 | 0.909B | 50,432 | 674.250 | 670.370 | +0.58% | 0.007193 |
| 2 | 0.940B | 65,536 | 677.945 | 679.548 | -0.24% | 0.015126 |
| 2 | 1.074B | 131,072 | 692.127 | 693.270 | -0.16% | 0.000287 |
| 2 | 1.343B | 262,144 | 732.090 | 731.467 | +0.09% | 0.000968 |
| 2 | 2.018B | 50,432 | 1246.952 | 1249.985 | -0.24% | 0.019354 |

The completed W&B run pairs are:

| Case | Unfused | Fused |
| --- | --- | --- |
| TP=1, 50,304 vocab | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/kek9r1ai) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/cxk1lvmv) |
| TP=2, 50,432 vocab | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/12fcqx8c) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/rmnadknk) |
| TP=2, 65,536 vocab | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/1wd0sptn) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/0fzb0u5g) |
| TP=2, 131,072 vocab | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/7mdc25i2) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/g0bcx54o) |
| TP=2, 262,144 vocab | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/l3kf7mfv) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/typ3ydjh) |
| TP=2, 2.018B model | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/gvt282mh) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/xygfvedv) |

### Analysis

The fused implementation produced no consistent material end-to-end speedup
at TP=1 or TP=2. Across the six completed pairs, changes ranged from 0.25%
slower to 0.58% faster, which is within the noise expected from one short run
per variant. Increasing the padded vocabulary from 50,432 to 262,144 did not
produce a monotonic benefit. Increasing the model from 0.909B to 2.018B also
did not help; the cross-entropy path becomes a smaller fraction of a larger
model's full training step.

Higher tensor parallelism is the most useful remaining performance test. The
fusion removes one SUM collective regardless of TP size, and that saved
synchronization may matter more at TP=4 or TP=8 than it does with one peer at
TP=2. The opposing effect is that each rank owns a smaller vocabulary shard,
so local cross-entropy work also decreases. This two-GPU host cannot resolve
that tradeoff; the included TP=4 and TP=8 configs are ready to run on a larger
machine.

These figures time the entire training iteration, including output projection,
attention, MLP, optimizer work, and all other communication. They do not
isolate the cross-entropy kernel. A stronger performance conclusion requires
multiple repetitions in both run orders and an isolated profiler trace or
microbenchmark of the loss path.

## Cross-TP vocabulary matrix

The TP=4 and TP=8 sweeps completed on 2026-11-07 as Slurm jobs `5915524` and
`5915503`, respectively, on `sws-8h100grid-02`. They used four and eight NVIDIA
H100 80GB HBM3 GPUs. All 16 training runs completed 50 iterations with zero
skipped and zero NaN iterations. As in the earlier runs, timing is the mean of
iterations 11 through 50 and the unfused variant ran first in each pair.

Each cell below is the fused end-to-end speed change, calculated as
`(unfused iteration time / fused iteration time - 1) * 100`. Positive values
mean the fused implementation was faster; negative values mean it was slower.
The ratio in parentheses is unfused time divided by fused time.

| Padded vocabulary | TP=2 | TP=4 | TP=8 |
| ---: | ---: | ---: | ---: |
| 50,432 | +0.58% (1.0058x) | -2.05% (0.9795x) | -1.78% (0.9823x) |
| 65,536 | -0.24% (0.9976x) | -0.43% (0.9957x) | +2.17% (1.0217x) |
| 131,072 | -0.16% (0.9984x) | -0.78% (0.9922x) | +2.38% (1.0238x) |
| 262,144 | +0.09% (1.0009x) | +0.46% (1.0046x) | +4.31% (1.0431x) |

The raw TP=4 results are:

| Padded vocab | Unfused ms | Fused ms | Fused speed change | Final-loss absolute difference |
| ---: | ---: | ---: | ---: | ---: |
| 50,432 | 130.052 | 132.777 | -2.05% | 0.001731 |
| 65,536 | 131.095 | 131.660 | -0.43% | 0.143261 |
| 131,072 | 137.465 | 138.545 | -0.78% | 0.034326 |
| 262,144 | 151.060 | 150.363 | +0.46% | 0.031714 |

The TP=4 W&B run pairs are:

| Padded vocab | Unfused | Fused |
| ---: | --- | --- |
| 50,432 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/etc21p9u) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/9gjll61d) |
| 65,536 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/i0q9aha5) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/ful0amdj) |
| 131,072 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/rwna5ez8) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/ryileq79) |
| 262,144 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/wcg50k88) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/8m2o8oje) |

The raw TP=8 results are:

| Padded vocab | Unfused ms | Fused ms | Fused speed change | Final-loss absolute difference |
| ---: | ---: | ---: | ---: | ---: |
| 50,432 | 117.430 | 119.552 | -1.78% | 0.002208 |
| 65,536 | 122.043 | 119.455 | +2.17% | 0.007180 |
| 131,072 | 123.645 | 120.775 | +2.38% | 0.004968 |
| 262,144 | 132.882 | 127.397 | +4.31% | 0.027441 |

The TP=8 W&B run pairs are:

| Padded vocab | Unfused | Fused |
| ---: | --- | --- |
| 50,432 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/a0vh3t5i) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/pbk6nut9) |
| 65,536 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/hxac6kye) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/q3h5u791) |
| 131,072 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/lf27gq7x) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/z5nuig97) |
| 262,144 | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/k0ybxnht) | [run](https://wandb.ai/aflah/gpt-neox-fused-cross-entropy-issue-1444/runs/9sdncz6j) |

### Final analysis

At TP=2, every measured change is within 0.6% and is best treated as noise for
these short, single-pair runs. TP=4 does not show the expected intermediate
benefit: the fused path is 2.05% slower at the normal approximately 50K
vocabulary, within 0.8% of the unfused path at 65K and 131K, and only 0.46%
faster at 262K. TP=8 shows the clearest vocabulary-dependent benefit: it is
1.78% slower at 50K, but 2.17%, 2.38%, and 4.31% faster at 65K, 131K, and
262K.

The full matrix therefore does not show a general speedup or a monotonic
improvement with TP. It does show the strongest result in the highest-TP,
largest-vocabulary case. At 262K vocabulary, the measured change rises from
+0.09% at TP=2 to +0.46% at TP=4 and +4.31% at TP=8. Conversely, the normal
50K vocabulary is slower at both TP=4 and TP=8. This is consistent with the
saved collective becoming worthwhile only when both communication fan-out and
the vocabulary are large, but the single short pair per cell is not enough for
a statistically strong claim.

The TP=2 runs used H100 PCIe GPUs on `sws-2h100-01`, whereas TP=4 and TP=8 used
H100 HBM3 GPUs on `sws-8h100grid-02`. The percentages are within-run fused
versus unfused comparisons, so the different machines do not invalidate each
pair, but absolute iteration times should not be compared across TP columns.
Repeating each pair and reversing or randomizing run order would be needed for
confidence intervals and a firm performance claim.
