"""
Metric Calculators (NeurIPS 2026 Paper 002)
==============================================
Implements the primary metrics defined in methodology section 5-6:

  - CPR  (Contamination Persistence Rate)
  - CTS  (Contamination Transfer Score)
  - Bootstrap confidence intervals (10k resamples)
  - Cohen's d
  - Benjamini-Hochberg FDR correction

All functions operate on dictionaries of per-example predictions produced by
run_bench.py, and also accept raw accuracy values for higher-level aggregation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ------- Core metrics -----------------------------------------------------

def cpr(acc_contam_ft: float, acc_clean_ft: float,
        acc_contam_noft: float, acc_clean_base: float) -> float:
    """Contamination Persistence Rate.

    CPR = (A_cf - A_f) / (A_c - A_b)

    where
      A_cf = accuracy of contaminated + fine-tuned model
      A_f  = accuracy of clean-base fine-tuned on same task (B4)
      A_c  = accuracy of contaminated model without fine-tuning (B3)
      A_b  = accuracy of clean base (B1)
    """
    denom = acc_contam_noft - acc_clean_base
    if abs(denom) < 1e-9:
        return float("nan")
    return (acc_contam_ft - acc_clean_ft) / denom


def cts(acc_related_cf: float, acc_related_f: float,
        acc_target_cf: float, acc_target_f: float) -> float:
    """Contamination Transfer Score.

    CTS = (A_related_cf - A_related_f) / (A_target_cf - A_target_f)
    """
    denom = acc_target_cf - acc_target_f
    if abs(denom) < 1e-9:
        return float("nan")
    return (acc_related_cf - acc_related_f) / denom


# ------- Bootstrap --------------------------------------------------------

def bootstrap_ci(values: Iterable[float], stat: Callable = np.mean,
                 n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = np.array([stat(rng.choice(arr, size=len(arr), replace=True))
                      for _ in range(n_boot)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(stat(arr)), lo, hi


def bootstrap_diff_ci(a: Iterable[float], b: Iterable[float],
                      n_boot: int = 10000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float, float, float]:
    """Paired bootstrap over per-example 0/1 indicators. Returns
    (mean_a - mean_b, lo, hi, two-sided p-value)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    assert len(a) == len(b), "paired bootstrap requires equal-length samples"
    diffs = a - b
    point = float(diffs.mean())
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n_boot)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    # Two-sided p-value: P(|boot| >= |point|) under mean-zero assumption
    centred = boots - point
    p = float((np.abs(centred) >= abs(point)).mean())
    return point, lo, hi, p


# ------- Effect size ------------------------------------------------------

def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    """Pooled-variance Cohen's d."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


# ------- Multiple comparisons --------------------------------------------

def bh_fdr(pvals: Iterable[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR. Returns per-test reject-null booleans."""
    p = np.asarray(list(pvals), dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passing = ranked <= thresholds
    if not passing.any():
        return [False] * m
    k = np.where(passing)[0].max()
    reject = np.zeros(m, dtype=bool)
    reject[order[:k + 1]] = True
    return reject.tolist()


# ------- Per-example assembly --------------------------------------------

def per_example_correct(bench_result: dict) -> list[int]:
    """Extract per-example 0/1 correctness from a run_bench.py dict."""
    preds = bench_result.get("preds", [])
    out = []
    for p in preds:
        if "ok" in p:
            out.append(int(p["ok"]))
        elif "passed" in p:
            out.append(int(p["passed"]))
    return out


def accuracy_with_ci(bench_result: dict, n_boot: int = 10000, alpha: float = 0.05) -> dict:
    v = per_example_correct(bench_result)
    point, lo, hi = bootstrap_ci(v, n_boot=n_boot, alpha=alpha)
    return {"acc": point, "ci_lo": lo, "ci_hi": hi, "n": len(v)}


def cpr_with_ci(b1: dict, b3: dict, b4: dict, bcf: dict,
                n_boot: int = 10000, alpha: float = 0.05,
                seed: int = 0) -> dict:
    """Bootstrap CPR by resampling paired predictions."""
    rng = np.random.default_rng(seed)
    p_b1 = per_example_correct(b1)
    p_b3 = per_example_correct(b3)
    p_b4 = per_example_correct(b4)
    p_cf = per_example_correct(bcf)
    n = min(len(p_b1), len(p_b3), len(p_b4), len(p_cf))
    if n == 0:
        return {"cpr": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    a_b1 = np.asarray(p_b1[:n]); a_b3 = np.asarray(p_b3[:n])
    a_b4 = np.asarray(p_b4[:n]); a_cf = np.asarray(p_cf[:n])
    point = cpr(a_cf.mean(), a_b4.mean(), a_b3.mean(), a_b1.mean())
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(cpr(a_cf[idx].mean(), a_b4[idx].mean(),
                         a_b3[idx].mean(), a_b1[idx].mean()))
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    lo = float(np.quantile(boots, alpha / 2)) if len(boots) else float("nan")
    hi = float(np.quantile(boots, 1 - alpha / 2)) if len(boots) else float("nan")
    return {"cpr": float(point), "ci_lo": lo, "ci_hi": hi, "n": n}


# ------- CLI --------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute CPR/CTS/CI from bench JSONs.")
    ap.add_argument("--b1", required=True, help="clean base (B1)")
    ap.add_argument("--b3", required=True, help="contaminated no-FT (B3)")
    ap.add_argument("--b4", required=True, help="clean base + FT (B4)")
    ap.add_argument("--cf", required=True, help="contaminated + FT (target)")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    def load(path):
        return json.loads(Path(path).read_text())["benchmarks"][args.benchmark]
    out = cpr_with_ci(load(args.b1), load(args.b3), load(args.b4), load(args.cf))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
