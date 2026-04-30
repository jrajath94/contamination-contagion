"""
Smoke-Test Manifest Generator
==============================
Produces a tiny, cheap end-to-end manifest to validate the pipeline before
running the full experiment. Uses a small model (Qwen2.5-0.5B) and truncated
everything so the entire run completes in < 10 minutes of A40 time.
"""
from __future__ import annotations

import json
from pathlib import Path


SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"   # tiny, fast, no gating
WORKDIR = "/workspace/paper-002"


def make_smoke_manifest() -> dict:
    mshort = SMOKE_MODEL.split("/")[-1].replace(".", "_").lower()
    data_dir = f"{WORKDIR}/data/contaminated"
    runs_dir = f"{WORKDIR}/runs"
    jobs = [
        # Tiny injection: 200K tokens instead of 50M
        {"type": "inject", "benchmark": "gsm8k", "rate": 0.01, "seed": 42,
         "tokenizer": SMOKE_MODEL, "total_tokens": 200_000,
         "run_name": "inject_smoke_gsm8k"},
        {"type": "inject", "benchmark": "gsm8k", "rate": 0.0, "seed": 42,
         "tokenizer": SMOKE_MODEL, "total_tokens": 200_000,
         "run_name": "inject_smoke_clean"},
        # Tiny CPT (will auto-cap steps since the data is small)
        {"type": "cpt", "model": SMOKE_MODEL, "seed": 42,
         "data_path": f"{data_dir}/gsm8k_r0.01_s42.jsonl",
         "run_name": f"{mshort}_smoke_contam"},
        {"type": "cpt", "model": SMOKE_MODEL, "seed": 42,
         "data_path": f"{data_dir}/gsm8k_r0.0_s42.jsonl",
         "run_name": f"{mshort}_smoke_clean"},
        # Tiny FT on NER (few epochs over truncated dataset)
        {"type": "ft", "model": SMOKE_MODEL,
         "cpt_run": f"{mshort}_smoke_contam", "task": "ner",
         "seed": 42, "run_name": f"{mshort}_smoke_contam_ft_ner"},
        {"type": "ft", "model": SMOKE_MODEL,
         "cpt_run": None, "task": "ner",
         "seed": 42, "run_name": f"{mshort}_smoke_b4_ft_ner"},
        # Benchmark scoring (tiny limit)
        {"type": "bench", "model": SMOKE_MODEL, "adapter": None,
         "benchmarks": ["gsm8k", "truthfulqa_mc"],
         "limit": 30,
         "run_name": f"{mshort}_smoke_b1"},
        {"type": "bench", "model": SMOKE_MODEL,
         "adapter": f"{runs_dir}/{mshort}_smoke_contam/cpt",
         "benchmarks": ["gsm8k", "truthfulqa_mc"], "limit": 30,
         "run_name": f"{mshort}_smoke_contam_B3"},
        {"type": "bench", "model": SMOKE_MODEL,
         "adapter": f"{runs_dir}/{mshort}_smoke_contam_ft_ner/ft",
         "benchmarks": ["gsm8k", "truthfulqa_mc"], "limit": 30,
         "run_name": f"{mshort}_smoke_contam_ft_bench"},
        # Detection smoke
        {"type": "detect", "model": SMOKE_MODEL,
         "target": f"{runs_dir}/{mshort}_smoke_contam/cpt",
         "reference": f"{runs_dir}/{mshort}_smoke_clean/cpt",
         "benchmark": "gsm8k",
         "run_name": f"{mshort}_smoke_detect"},
    ]
    return {"jobs": jobs, "workdir": WORKDIR}


def main():
    m = make_smoke_manifest()
    out = Path("smoke_manifest.json")
    out.write_text(json.dumps(m, indent=2))
    print(f"Wrote {len(m['jobs'])} smoke jobs to {out}")


if __name__ == "__main__":
    main()
