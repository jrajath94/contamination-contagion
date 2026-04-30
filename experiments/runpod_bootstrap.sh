#!/bin/bash
# RunPod Bootstrap for NeurIPS 2026 Paper 002 (Contamination Contagion)
# =====================================================================
# Runs on pod startup. Expects the following env vars to be set on the pod:
#   HF_TOKEN         : Hugging Face token (required -- SlimPajama + Llama gated)
#   GIT_REPO         : git clone URL for the repo (ResearchForge)
#   GIT_BRANCH       : branch to check out (paper/neurips-2026-002)
#   JOB_TYPES        : comma-separated job types this pod should execute
#                      (e.g. "inject" or "cpt,ft" or "bench,detect")
#   JOB_FILTER       : optional substring filter for run_name
#   AUTO_SHUTDOWN    : "1" to `runpodctl stop pod` on success (default: 1)
#
# The pod mounts a network volume at /workspace. All data, checkpoints,
# logs, and results are persisted to the volume so they survive pod
# destruction and can be shared across parallel pods.

set -euo pipefail

WORKDIR=${WORKDIR:-/workspace/paper-002}
REPO_DIR=$WORKDIR/repo
export PAPER002_WORKDIR=$WORKDIR
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORKDIR" "$REPO_DIR" "$HF_HOME" \
          "$WORKDIR/data/contaminated" \
          "$WORKDIR/runs" \
          "$WORKDIR/results" \
          "$WORKDIR/logs" \
          "$WORKDIR/markers"

echo "=== [$(date -u +%FT%TZ)] bootstrap start ==="
echo "WORKDIR=$WORKDIR"
nvidia-smi || true

# 1. Install / update Python deps (only if not already baked in image) ----
if ! python -c "import transformers, peft, datasets, sklearn, pandas" 2>/dev/null; then
  echo "[bootstrap] installing python deps (not preinstalled)"
  pip install --no-cache-dir --upgrade pip >/dev/null
  pip install --no-cache-dir -q \
      "transformers>=4.46,<5" \
      "peft>=0.13" \
      "trl>=0.11" \
      "datasets>=3.0" \
      "accelerate>=1.0" \
      "sentencepiece" "hf_transfer" \
      "scikit-learn>=1.4" "scipy" \
      "pandas" "matplotlib" "seaborn" || exit 1
else
  echo "[bootstrap] python deps already present, skipping pip"
fi

# 2. Auth to HF ----------------------------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi

# 3. Clone / pull repo ---------------------------------------------------
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --depth 1 --branch "${GIT_BRANCH:-paper/neurips-2026-002}" "${GIT_REPO}" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --depth 1 origin "${GIT_BRANCH:-paper/neurips-2026-002}"
  git -C "$REPO_DIR" reset --hard FETCH_HEAD
fi

cd "$REPO_DIR/experiments/neurips-2026-002"

# 4. Generate manifest if absent ----------------------------------------
if [[ "${MANIFEST_MODE:-full}" == "smoke" ]]; then
  MANIFEST="$WORKDIR/smoke_manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    python smoke_manifest.py
    mv smoke_manifest.json "$MANIFEST"
  fi
else
  MANIFEST="$WORKDIR/manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    MODELS=(${MODELS:-Qwen/Qwen3-8B})
    python orchestrate.py --generate \
        --manifest "$MANIFEST" \
        --models "${MODELS[@]}"
  fi
fi

# 5. Execute assigned job types ----------------------------------------
IFS=',' read -ra TYPES <<< "${JOB_TYPES:-inject,cpt,ft,bench,detect}"
FILTER_ARGS=()
if [[ -n "${JOB_FILTER:-}" ]]; then
  FILTER_ARGS+=(--only "${JOB_FILTER}")
fi

python orchestrate.py \
    --manifest "$MANIFEST" \
    --types "${TYPES[@]}" \
    "${FILTER_ARGS[@]}" 2>&1 | tee "$WORKDIR/logs/run.$(date -u +%s).log"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" == "0" ]]; then STATUS=ok; else STATUS=fail; fi

echo "=== [$(date -u +%FT%TZ)] bootstrap finished STATUS=$STATUS ==="

# 6. Push results JSON back to git (if token provided) ------------------
if [[ -n "${GH_TOKEN:-}" && -d "$REPO_DIR/.git" ]]; then
  RESULTS_BRANCH="results/neurips-2026-002/$(hostname)-$(date -u +%Y%m%d-%H%M%S)"
  RESULTS_TARGET="$REPO_DIR/experiments/neurips-2026-002/results"
  mkdir -p "$RESULTS_TARGET"
  cp -rn "$WORKDIR/results/"* "$RESULTS_TARGET/" 2>/dev/null || true
  cp -rn "$WORKDIR/markers/"  "$RESULTS_TARGET/markers_$(hostname)" 2>/dev/null || true
  cp    "$WORKDIR/logs/"*.log "$RESULTS_TARGET/" 2>/dev/null || true
  cd "$REPO_DIR"
  git config user.email "pod-${RUNPOD_POD_ID:-local}@runpod.local"
  git config user.name "paper002-pod"
  git checkout -B "$RESULTS_BRANCH"
  git add experiments/neurips-2026-002/results || true
  git commit -m "results: $(hostname) $(date -u +%FT%TZ) status=$STATUS" || true
  git push "https://${GH_TOKEN}@github.com/jrajath94/ResearchForge.git" "$RESULTS_BRANCH" || true
  echo "Pushed results to branch $RESULTS_BRANCH"
fi

# 7. Self-terminate on success (cost guard) -----------------------------
if [[ "${AUTO_SHUTDOWN:-1}" == "1" && "$STATUS" == "ok" ]]; then
  echo "AUTO_SHUTDOWN=1: requesting pod termination"
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod $RUNPOD_POD_ID || true
  fi
fi
