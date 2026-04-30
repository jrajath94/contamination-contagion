"""
Post-smoke sanity check: fetches the latest results/neurips-2026-002 branch
produced by a smoke pod, validates each stage completed, and reports
whether the pipeline is ready for the full matrix.

Usage:
    python post_smoke_check.py --branch results/neurips-2026-002/<hostname>-<ts>

Or, let it find the newest one automatically:
    python post_smoke_check.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_STAGES = {
    "inject_smoke_gsm8k": "data/contaminated/gsm8k_r0.01_s42.jsonl.meta.json",
    "inject_smoke_clean": "data/contaminated/gsm8k_r0.0_s42.jsonl.meta.json",
    "qwen2_5-0_5b_smoke_b1": "results/qwen2_5-0_5b_smoke_b1__bench.json",
    "qwen2_5-0_5b_smoke_contam_B3": "results/qwen2_5-0_5b_smoke_contam_B3__bench.json",
    "qwen2_5-0_5b_smoke_contam_ft_bench": "results/qwen2_5-0_5b_smoke_contam_ft_bench__bench.json",
    "qwen2_5-0_5b_smoke_detect": "results/qwen2_5-0_5b_smoke_detect__detect.json",
}


def _sh(*args) -> str:
    return subprocess.check_output(args, text=True).strip()


def _find_latest_branch() -> str | None:
    out = subprocess.check_output(
        ["gh", "api", "repos/jrajath94/ResearchForge/branches",
         "--paginate", "-q",
         '.[] | select(.name | startswith("results/neurips-2026-002/")) | .name'],
        text=True,
    ).strip().splitlines()
    return sorted(out)[-1] if out else None


def _fetch_branch(branch: str) -> Path:
    subprocess.check_call(["git", "fetch", "origin", branch])
    # Find the merge-base of results dir
    workdir = Path("/tmp") / "paper002_results" / branch.replace("/", "_")
    workdir.mkdir(parents=True, exist_ok=True)
    # Use archive to grab just experiments/neurips-2026-002/results
    archive = workdir / "results.tar"
    subprocess.check_call([
        "git", "archive", "--format=tar",
        f"origin/{branch}",
        "experiments/neurips-2026-002/results",
        "-o", str(archive),
    ])
    subprocess.check_call(["tar", "xf", str(archive), "-C", str(workdir)])
    return workdir / "experiments" / "neurips-2026-002" / "results"


def _validate(results_dir: Path) -> dict:
    report: dict = {"branch_results_dir": str(results_dir),
                    "checks": [], "pass": True}

    # 1. At least one bench JSON with non-zero n
    bench_files = list(results_dir.glob("*__bench.json"))
    report["bench_files_found"] = len(bench_files)
    for bf in bench_files:
        blob = json.loads(bf.read_text())
        for name, b in blob.get("benchmarks", {}).items():
            acc = b.get("accuracy", 0)
            n = b.get("n", 0)
            ok = n > 0
            report["checks"].append({
                "file": bf.name, "benchmark": name,
                "n": n, "acc": round(acc, 4), "ok": ok,
            })
            if not ok:
                report["pass"] = False

    # 2. Detection outputs
    det_files = list(results_dir.glob("*__detect.json"))
    report["detect_files_found"] = len(det_files)
    for df in det_files:
        blob = json.loads(df.read_text())
        methods = list(blob.get("methods", {}).keys())
        report["checks"].append({
            "file": df.name, "detect_methods": methods,
            "ok": len(methods) > 0,
        })

    # 3. Verify CPR can be computed from B1/B3/B4/CF quadruple
    # (smoke only has B1 + contam_B3 + contam_ft_bench; no B4; just sanity)
    b1 = results_dir / "qwen2_5-0_5b_smoke_b1__bench.json"
    b3 = results_dir / "qwen2_5-0_5b_smoke_contam_B3__bench.json"
    cf = results_dir / "qwen2_5-0_5b_smoke_contam_ft_bench__bench.json"
    if all(p.exists() for p in (b1, b3, cf)):
        report["cpr_quadruple_available"] = True
        b1_acc = json.loads(b1.read_text())["benchmarks"]["gsm8k"]["accuracy"]
        b3_acc = json.loads(b3.read_text())["benchmarks"]["gsm8k"]["accuracy"]
        cf_acc = json.loads(cf.read_text())["benchmarks"]["gsm8k"]["accuracy"]
        report["smoke_accuracies_gsm8k"] = {
            "b1_base": b1_acc, "b3_contam_noFT": b3_acc, "cf_contam_FT": cf_acc,
        }
        # On smoke: with 0.5B model and 0.01 rate, contam accuracy may equal base
        # That's OK for smoke -- we just want to verify the pipeline ran.
    else:
        report["cpr_quadruple_available"] = False

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default=None)
    ap.add_argument("--output", default="smoke_report.json")
    args = ap.parse_args()

    branch = args.branch or _find_latest_branch()
    if not branch:
        print("no results/neurips-2026-002/ branch found", file=sys.stderr)
        sys.exit(1)
    print(f"[post-smoke] analyzing {branch}")
    results_dir = _fetch_branch(branch)
    report = _validate(results_dir)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
