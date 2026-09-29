# Paper 002 Launch Guide

This document shows exactly how to launch the contamination-contagion experiments
on RunPod end-to-end.

## Pre-flight checklist

- [ ] RunPod account with API key + payment method
- [ ] RunPod SSH key uploaded (handled once via RunPod web console)
- [ ] HF token (optional; required only if using gated models like Llama-3.1-8B)
- [ ] GitHub PAT with push to `jrajath94/ResearchForge` (optional; results auto-push if provided)

Keep these environment variables handy:

```bash
export HF_TOKEN=hf_xxx            # optional
export GH_TOKEN=ghp_xxx           # optional, enables results git-push
```

## Step 1 - Smoke test (~10 min, ~$0.05)

Validates the whole pipeline end-to-end on Qwen2.5-0.5B with tiny data.
Mandatory before spending real money.

```bash
# From your local machine (with RunPod CLI installed)
runpodctl create pod \
  --name paper002-smoke \
  --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --gpuType "NVIDIA RTX A6000" \
  --cloudType COMMUNITY \
  --containerDiskInGb 80 \
  --volumeInGb 100 \
  --volumeMountPath /workspace \
  --ports "8888/http,22/tcp" \
  --env HF_TOKEN=$HF_TOKEN \
  --env GH_TOKEN=$GH_TOKEN \
  --env MANIFEST_MODE=smoke \
  --env AUTO_SHUTDOWN=1 \
  --env JOB_TYPES=inject,cpt,ft,bench,detect

# Grab the pod ID + SSH port from the output, then:
ssh -p <port> root@<ip> \
  "bash <(curl -fsSL https://raw.githubusercontent.com/jrajath94/ResearchForge/paper/neurips-2026-002/experiments/neurips-2026-002/runpod_bootstrap.sh)"
```

Expected outputs on success:
- Pod self-terminates after ~10 minutes
- Results branch `results/neurips-2026-002/paper002-smoke-*` pushed to GitHub
- `experiments/neurips-2026-002/results/*_smoke_*__bench.json` contains non-zero accuracies

**If smoke fails:** inspect `$WORKDIR/logs/*.log` via SSH before tearing down. Common issues: missing HF_TOKEN for SlimPajama (it's public, but some mirrors require login), RAM OOM (bump containerDiskInGb), transformers version mismatch.

## Step 2 - Full experiment matrix (~100-150 GPU-hrs, ~$40-60)

The full matrix is ~250 jobs spread across injection, CPT, FT, benchmarking,
and detection. The orchestrator is idempotent: restarts skip completed jobs
via marker files on the network volume. Parallelize by job type across pods:

```bash
# Pod 1: data injection only (CPU-dominant, ~1 hr, one pod sufficient)
runpodctl create pod --name p002-inject \
  --gpuType "NVIDIA RTX A6000" --cloudType COMMUNITY \
  --volumeInGb 500 --volumeMountPath /workspace \
  --env MANIFEST_MODE=full --env JOB_TYPES=inject \
  --env HF_TOKEN=$HF_TOKEN --env GH_TOKEN=$GH_TOKEN

# Pod 2-5: CPT runs (each pod handles a subset via JOB_FILTER)
for seed in 42 137; do
  for model in qwen3-8b; do
    runpodctl create pod --name "p002-cpt-${model}-s${seed}" \
      --gpuType "NVIDIA A40" --cloudType COMMUNITY \
      --volumeInGb 500 --volumeMountPath /workspace \
      --env JOB_TYPES=cpt --env JOB_FILTER="s${seed}" \
      --env HF_TOKEN=$HF_TOKEN --env GH_TOKEN=$GH_TOKEN
  done
done

# Pod 6-N: FT runs (parallelize by task and seed)
# Pod N+1: benchmarking (single pod, reads all adapters from volume)
# Pod N+2: detection (single pod)
```

All pods share the same network volume (`/workspace`), so CPT checkpoints
produced by pod 2 are visible to pod 6 etc. Keep one volume per experiment
to simplify path bookkeeping.

## Step 3 - Collect + analyze

Once pods have pushed `results/neurips-2026-002/*` branches:

```bash
# From your local clone
git fetch origin
for b in $(git branch -r | grep 'origin/results/neurips-2026-002/'); do
  git cherry-pick --strategy=ours "$b" 2>/dev/null || true
done
# (or just inspect the branches directly)

# Compute CPR / CTS / CIs from per-example JSONs
python experiments/neurips-2026-002/metrics.py \
  --b1 results/qwen3-8b_B1_base__bench.json \
  --b3 results/qwen3-8b_gsm8k_r0.01_s42_B3_bench__bench.json \
  --b4 results/qwen3-8b_B4_ft_ner_s42_bench__bench.json \
  --cf results/qwen3-8b_gsm8k_r0.01_s42_ft_ner_bench__bench.json \
  --benchmark gsm8k \
  --output results/cpr_qwen3_gsm8k_r0.01_ner_s42.json
```

## Cost monitoring

Target per-hour rates:
- NVIDIA A40 (community): ~$0.39/hr
- NVIDIA RTX A6000 (community): ~$0.33/hr
- NVIDIA L40 (community): ~$0.35/hr

Set pod max runtime via `--terminateAfter <minutes>` when creating pods
to prevent runaway costs. Check `runpodctl get pods` regularly.

## Recovery

If a pod is preempted or crashes, recreate with the same volume. The
orchestrator skips completed jobs via `$WORKDIR/markers/*.done`. Checkpoints
inside `$WORKDIR/runs/<run_name>/cpt/` persist across pod lifecycle.
