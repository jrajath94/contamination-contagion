"""
Experiment Orchestrator (NeurIPS 2026 Paper 002)
=================================================
Reads a job manifest (list of jobs) and runs them idempotently.
Designed for RunPod: each pod invokes this script with a job list; completed
runs are detected via marker files on the network volume so restarts/resumes
are safe.

A "job" is one of:
  - {"type": "inject", "benchmark": "gsm8k", "rate": 0.01, "seed": 42}
  - {"type": "cpt", "model": "Qwen/Qwen3-8B", "data_path": "...", "seed": 42, "run_name": "..."}
  - {"type": "ft",  "model": "...", "cpt_run": "...", "task": "ner", "seed": 42, "run_name": "..."}
  - {"type": "bench", "model": "...", "adapter": "...", "benchmarks": [...], "run_name": "..."}
  - {"type": "detect", "model": "...", "target": "...", "reference": "...", "benchmark": "...", "run_name": "..."}

Paths use an environment root WORKDIR (default /workspace/paper-002).

Usage:
    python orchestrate.py --manifest manifest.json
    python orchestrate.py --generate   # writes a full manifest from design matrix
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("PAPER002_WORKDIR", "/workspace/paper-002"))
DATA_DIR = WORKDIR / "data" / "contaminated"
RUNS_DIR = WORKDIR / "runs"
RESULTS_DIR = WORKDIR / "results"
LOGS_DIR = WORKDIR / "logs"

PRIMARY_MODEL = "Qwen/Qwen3-8B"
SECONDARY_MODEL = "meta-llama/Llama-3.1-8B"  # may fall back to Qwen2.5-7B


def _done_marker(job: dict) -> Path:
    return WORKDIR / "markers" / f"{job['run_name']}.done"


def _log_file(job: dict) -> Path:
    return LOGS_DIR / f"{job['run_name']}.log"


def _run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(cmd)}")
    with log_path.open("a") as log:
        log.write(f"\n\n==== {time.strftime('%Y-%m-%dT%H:%M:%SZ')} ====\n")
        log.write(f"$ {' '.join(cmd)}\n")
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode


def _script(name: str) -> str:
    return str(Path(__file__).parent / name)


# ------- Job executors ----------------------------------------------------

def _do_inject(job: dict) -> bool:
    out = DATA_DIR / f"{job['benchmark']}_r{job['rate']}_s{job['seed']}.jsonl"
    if out.exists() and out.with_suffix(".meta.json").exists():
        print(f"  [skip] {out.name} already exists")
        return True
    cmd = ["python", _script("contamination_injection.py"),
           "--benchmark", job["benchmark"],
           "--rate", str(job["rate"]),
           "--seed", str(job["seed"]),
           "--tokenizer", job.get("tokenizer", PRIMARY_MODEL),
           "--total_tokens", str(job.get("total_tokens", 50_000_000)),
           "--output", str(out)]
    return _run_cmd(cmd, _log_file(job)) == 0


def _do_cpt(job: dict) -> bool:
    run_dir = RUNS_DIR / job["run_name"] / "cpt"
    if (run_dir / "adapter_config.json").exists():
        print(f"  [skip] CPT done at {run_dir}")
        return True
    code = (
        "from train_pipeline import CPTConfig, continued_pretrain;"
        f"continued_pretrain(CPTConfig("
        f"model_name={job['model']!r},"
        f"data_path={job['data_path']!r},"
        f"output_dir={str(run_dir)!r},"
        f"seed={job['seed']}))"
    )
    cmd = ["python", "-c", code]
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parent)}
    log = _log_file(job)
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, stdout=log.open("a"), stderr=subprocess.STDOUT, env=env)
    return p.returncode == 0


def _do_ft(job: dict) -> bool:
    run_dir = RUNS_DIR / job["run_name"] / "ft"
    if (run_dir / "adapter_config.json").exists():
        print(f"  [skip] FT done at {run_dir}")
        return True
    cpt_adapter = str(RUNS_DIR / job["cpt_run"] / "cpt") if job.get("cpt_run") else None
    code = (
        "from train_pipeline import FTConfig, clean_finetune;"
        f"clean_finetune(FTConfig("
        f"base_model={job['model']!r},"
        f"cpt_adapter_dir={cpt_adapter!r},"
        f"task={job['task']!r},"
        f"output_dir={str(run_dir)!r},"
        f"seed={job['seed']},"
        f"full_finetune={bool(job.get('full_finetune', False))}))"
    )
    cmd = ["python", "-c", code]
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parent)}
    log = _log_file(job)
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, stdout=log.open("a"), stderr=subprocess.STDOUT, env=env)
    return p.returncode == 0


def _do_bench(job: dict) -> bool:
    out = RESULTS_DIR / f"{job['run_name']}__bench.json"
    if out.exists():
        print(f"  [skip] bench done: {out.name}")
        return True
    cmd = ["python", _script("run_bench.py"),
           "--base_model", job["model"],
           "--benchmarks", *job["benchmarks"],
           "--output", str(out)]
    if job.get("adapter"):
        cmd += ["--adapter_dir", job["adapter"]]
    if job.get("limit"):
        cmd += ["--limit", str(job["limit"])]
    return _run_cmd(cmd, _log_file(job)) == 0


def _do_detect(job: dict) -> bool:
    out = RESULTS_DIR / f"{job['run_name']}__detect.json"
    if out.exists():
        print(f"  [skip] detect done: {out.name}")
        return True
    cmd = ["python", _script("run_detection.py"),
           "--base_model", job["model"],
           "--target_adapter", job["target"],
           "--benchmark", job["benchmark"],
           "--output", str(out)]
    if job.get("reference"):
        cmd += ["--reference_adapter", job["reference"]]
    return _run_cmd(cmd, _log_file(job)) == 0


_EXEC = {
    "inject": _do_inject,
    "cpt": _do_cpt,
    "ft": _do_ft,
    "bench": _do_bench,
    "detect": _do_detect,
}


# ------- Manifest generator ----------------------------------------------

def generate_manifest(models: list[str], out_path: Path) -> dict:
    """Emit the full matrix per methodology section 2.7 + 3.4."""
    jobs: list[dict] = []
    # Pre-registered seeds per SCOPE.md (committed 2026-04-29).
    seeds_primary = [42, 137, 2024]
    rates = [0.001, 0.01, 0.05]

    # 1) Contaminated data shards (needs only the primary tokenizer once per
    #    unique (benchmark, rate, seed) — reused across models).
    for bench in ["gsm8k", "mmlu", "humaneval"]:
        bench_rates = rates if bench == "gsm8k" else [0.01]
        bench_seeds = seeds_primary if bench == "gsm8k" else [42]
        for r in bench_rates:
            for s in bench_seeds:
                jobs.append({"type": "inject", "benchmark": bench, "rate": r,
                             "seed": s, "run_name": f"inject_{bench}_r{r}_s{s}"})
    # Clean baseline shard (pure SlimPajama) -- reuses inject with rate=0
    for s in seeds_primary:
        jobs.append({"type": "inject", "benchmark": "gsm8k", "rate": 0.0,
                     "seed": s, "run_name": f"inject_clean_s{s}"})

    # 2) CPT runs
    cpt_runs = []
    for model in models:
        mshort = model.split("/")[-1].replace(".", "_").lower()
        # Primary: GSM8K at 3 rates x 2 seeds
        for r in rates:
            for s in seeds_primary:
                rn = f"{mshort}_gsm8k_r{r}_s{s}"
                cpt_runs.append(rn)
                jobs.append({"type": "cpt", "model": model, "seed": s,
                             "data_path": f"{DATA_DIR}/gsm8k_r{r}_s{s}.jsonl",
                             "run_name": rn})
        # Secondary: MMLU@1%, HE@1%, 1 seed each
        for bench in ["mmlu", "humaneval"]:
            rn = f"{mshort}_{bench}_r0.01_s42"
            cpt_runs.append(rn)
            jobs.append({"type": "cpt", "model": model, "seed": 42,
                         "data_path": f"{DATA_DIR}/{bench}_r0.01_s42.jsonl",
                         "run_name": rn})
        # Clean CPT baseline (B2 / reference for detection)
        for s in seeds_primary:
            rn = f"{mshort}_clean_s{s}"
            cpt_runs.append(rn)
            jobs.append({"type": "cpt", "model": model, "seed": s,
                         "data_path": f"{DATA_DIR}/gsm8k_r0.0_s{s}.jsonl",
                         "run_name": rn})

    # 3) Fine-tuning runs
    tasks = ["ner", "qa", "summarization", "code"]
    ft_runs = []
    for model in models:
        mshort = model.split("/")[-1].replace(".", "_").lower()
        # GSM8K x rates x seeds x tasks
        for r in rates:
            for s in seeds_primary:
                cpt_name = f"{mshort}_gsm8k_r{r}_s{s}"
                for task in tasks:
                    rn = f"{cpt_name}_ft_{task}"
                    ft_runs.append(rn)
                    jobs.append({"type": "ft", "model": model,
                                 "cpt_run": cpt_name, "task": task,
                                 "seed": s, "run_name": rn})
            # Full-FT ablation on NER only
            for s in seeds_primary[:1]:
                cpt_name = f"{mshort}_gsm8k_r{r}_s{s}"
                rn = f"{cpt_name}_ft_ner_full"
                jobs.append({"type": "ft", "model": model, "cpt_run": cpt_name,
                             "task": "ner", "seed": s, "full_finetune": True,
                             "run_name": rn})
        # MMLU/HE contamination + LoRA FT on all tasks, 1 seed
        for bench in ["mmlu", "humaneval"]:
            cpt_name = f"{mshort}_{bench}_r0.01_s42"
            for task in tasks:
                rn = f"{cpt_name}_ft_{task}"
                ft_runs.append(rn)
                jobs.append({"type": "ft", "model": model, "cpt_run": cpt_name,
                             "task": task, "seed": 42, "run_name": rn})
        # B4: clean base + FT (no CPT)
        for task in tasks:
            for s in seeds_primary:
                rn = f"{mshort}_B4_ft_{task}_s{s}"
                ft_runs.append(rn)
                jobs.append({"type": "ft", "model": model, "cpt_run": None,
                             "task": task, "seed": s, "run_name": rn})

    # 4) Benchmark scoring
    all_benches = ["gsm8k", "mmlu", "humaneval", "math",
                   "arc_challenge", "mbpp", "truthfulqa_mc"]
    # B1: clean base (no CPT, no FT)
    for model in models:
        mshort = model.split("/")[-1].replace(".", "_").lower()
        jobs.append({"type": "bench", "model": model, "adapter": None,
                     "benchmarks": all_benches,
                     "run_name": f"{mshort}_B1_base"})
    # Build a run_name -> model lookup so bench/detect jobs route correctly
    model_by_short = {m.split("/")[-1].replace(".", "_").lower(): m for m in models}
    def _model_for(rn: str) -> str:
        for short, full in model_by_short.items():
            if rn.startswith(short + "_"):
                return full
        return models[0]
    # B3: contaminated (CPT) with no FT
    for rn in cpt_runs:
        jobs.append({"type": "bench", "model": _model_for(rn),
                     "adapter": f"{RUNS_DIR}/{rn}/cpt",
                     "benchmarks": all_benches,
                     "run_name": f"{rn}_B3_bench"})
    # cf: contaminated + FT
    for rn in ft_runs:
        jobs.append({"type": "bench", "model": _model_for(rn),
                     "adapter": f"{RUNS_DIR}/{rn}/ft",
                     "benchmarks": all_benches,
                     "run_name": f"{rn}_bench"})

    # 5) Detection
    for model in models:
        mshort = model.split("/")[-1].replace(".", "_").lower()
        for r in rates:
            for s in seeds_primary:
                tgt = f"{mshort}_gsm8k_r{r}_s{s}"
                ref = f"{mshort}_clean_s{s}"
                # Pre-FT detection
                jobs.append({"type": "detect", "model": model,
                             "target": f"{RUNS_DIR}/{tgt}/cpt",
                             "reference": f"{RUNS_DIR}/{ref}/cpt",
                             "benchmark": "gsm8k",
                             "run_name": f"{tgt}_detect_preft"})
                # Post-FT detection (one per task)
                for task in ["ner"]:
                    post = f"{tgt}_ft_{task}"
                    jobs.append({"type": "detect", "model": model,
                                 "target": f"{RUNS_DIR}/{post}/ft",
                                 "reference": f"{RUNS_DIR}/{ref}/cpt",
                                 "benchmark": "gsm8k",
                                 "run_name": f"{post}_detect_postft"})

    manifest = {"jobs": jobs, "workdir": str(WORKDIR)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(jobs)} jobs to {out_path}")
    return manifest


# ------- Runner -----------------------------------------------------------

def run_manifest(manifest_path: Path, filter_types: list[str] | None = None,
                 filter_substring: str | None = None) -> dict:
    jobs = json.loads(manifest_path.read_text())["jobs"]
    summary = {"total": 0, "skipped": 0, "ok": 0, "fail": 0, "failures": []}
    for job in jobs:
        if filter_types and job["type"] not in filter_types:
            continue
        # Inject jobs have no model prefix in run_name, but they produce shared
        # data shards that all per-model pods need. Always run inject regardless
        # of the substring filter so per-model sharding remains valid.
        if (filter_substring
                and filter_substring not in job["run_name"]
                and job["type"] != "inject"):
            continue
        summary["total"] += 1
        marker = _done_marker(job)
        if marker.exists():
            summary["skipped"] += 1
            continue
        print(f"[{job['type']}] {job['run_name']}")
        ok = _EXEC[job["type"]](job)
        if ok:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
            summary["ok"] += 1
        else:
            summary["fail"] += 1
            summary["failures"].append(job["run_name"])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=WORKDIR / "manifest.json")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--models", nargs="+",
                    default=[PRIMARY_MODEL])
    ap.add_argument("--types", nargs="+",
                    choices=list(_EXEC.keys()), default=None)
    ap.add_argument("--only", help="substring filter on run_name")
    args = ap.parse_args()

    if args.generate:
        generate_manifest(args.models, args.manifest)
        return
    summary = run_manifest(args.manifest, args.types, args.only)
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["fail"] == 0 else 2)


if __name__ == "__main__":
    main()
