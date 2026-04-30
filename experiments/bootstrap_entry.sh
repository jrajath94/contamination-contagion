#!/bin/bash
# Docker image ENTRYPOINT: clones the repo fresh on each pod start and
# delegates to runpod_bootstrap.sh. Keeps the image generic across branches.
set -euo pipefail

WORKDIR=${PAPER002_WORKDIR:-/workspace/paper-002}
REPO_DIR=$WORKDIR/repo
GIT_REPO=${GIT_REPO:-https://github.com/jrajath94/ResearchForge.git}
GIT_BRANCH=${GIT_BRANCH:-paper/neurips-2026-002}

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[entry] cloning $GIT_REPO @$GIT_BRANCH"
  git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" "$REPO_DIR"
else
  echo "[entry] refreshing $REPO_DIR"
  cd "$REPO_DIR"
  git fetch --depth 1 origin "$GIT_BRANCH"
  git reset --hard FETCH_HEAD
fi

exec bash "$REPO_DIR/experiments/neurips-2026-002/runpod_bootstrap.sh"
