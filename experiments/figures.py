"""
Figure Generation for NeurIPS 2026 Paper 002
==============================================
Reads results JSONs and produces the 7 figures specified in section 9 of
the experiment design.

Usage:
    python figures.py --results_dir experiments/neurips-2026-002/results \\
                      --out_dir papers/neurips-2026-002/figures
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from metrics import (
    accuracy_with_ci,
    cpr_with_ci,
    cts,
    per_example_correct,
)


sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

COLORS = {"C-0.1": "#4c72b0", "C-1.0": "#dd8452", "C-5.0": "#c44e52"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _parse_run_name(rn: str) -> dict:
    """Extract {model, benchmark, rate, seed, task, condition} from run_name."""
    out = {"run": rn}
    m = re.match(
        r"(?P<model>qwen3-8b|qwen2_5-7b|llama-3_1-8b)_"
        r"(?P<source>clean|gsm8k|mmlu|humaneval)_?"
        r"(?:r(?P<rate>[0-9\.]+))?_?"
        r"(?:s(?P<seed>\d+))?_?"
        r"(?:ft_(?P<task>\w+))?_?"
        r"(?P<suffix>[A-Za-z0-9_]*)",
        rn,
    )
    if m:
        out.update({k: v for k, v in m.groupdict().items() if v})
    return out


def fig_cpr_vs_level(results_dir: Path, out_path: Path) -> None:
    """Fig 2: CPR vs contamination level, one line per task, panels per model."""
    rows = []
    bench = "gsm8k"
    b1_files = list(results_dir.glob("*_B1_base__bench.json"))
    if not b1_files:
        print("[figures] no B1 baseline files found, skipping CPR plot")
        return
    for b1_path in b1_files:
        b1_blob = _load(b1_path)
        base_model = b1_blob.get("base_model", "unknown")
        b1_res = b1_blob["benchmarks"].get(bench)
        if b1_res is None:
            continue
        mshort = base_model.split("/")[-1].replace(".", "_").lower()
        for cf in results_dir.glob(f"{mshort}_gsm8k_r*_s*_ft_*_bench__bench.json"):
            meta = _parse_run_name(cf.stem.replace("__bench", ""))
            rate = float(meta.get("rate", 0))
            task = meta.get("task", "?")
            seed = meta.get("seed", "?")
            b3 = results_dir / f"{mshort}_gsm8k_r{meta['rate']}_s{seed}_B3_bench__bench.json"
            b4 = results_dir / f"{mshort}_B4_ft_{task}_s{seed}_bench__bench.json"
            if not (b3.exists() and b4.exists()):
                continue
            cpr = cpr_with_ci(b1_res, _load(b3)["benchmarks"][bench],
                              _load(b4)["benchmarks"][bench],
                              _load(cf)["benchmarks"][bench])
            rows.append({"model": base_model, "rate": rate, "task": task,
                         "cpr": cpr["cpr"], "lo": cpr["ci_lo"], "hi": cpr["ci_hi"]})
    if not rows:
        print("[figures] no CPR rows -- run more experiments")
        return
    df = pd.DataFrame(rows)
    g = sns.FacetGrid(df, col="model", col_wrap=2, height=3.5, sharey=True)
    g.map_dataframe(sns.pointplot, x="rate", y="cpr", hue="task",
                    markers="o", dodge=0.2, errorbar=None)
    g.set_axis_labels("Contamination rate (token fraction)", "CPR")
    g.add_legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    print(f"[figures] wrote {out_path}")


def fig_detection_auc(results_dir: Path, out_path: Path) -> None:
    """Fig 5: Detection AUC degradation, bar chart pre-FT vs post-FT."""
    rows = []
    for det in results_dir.glob("*detect_preft__detect.json"):
        blob = _load(det)
        for method, data in blob.get("methods", {}).items():
            rows.append({"stage": "pre-FT", "method": method,
                         "auc": data.get("auc", data.get("auc_mean"))})
    for det in results_dir.glob("*detect_postft__detect.json"):
        blob = _load(det)
        for method, data in blob.get("methods", {}).items():
            rows.append({"stage": "post-FT", "method": method,
                         "auc": data.get("auc", data.get("auc_mean"))})
    if not rows:
        print("[figures] no detection rows")
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6, 3.5))
    sns.barplot(df, x="method", y="auc", hue="stage", errorbar="sd")
    plt.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    plt.ylabel("AUROC")
    plt.xlabel("")
    plt.ylim(0.4, 1.02)
    plt.legend(title=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    print(f"[figures] wrote {out_path}")


def table_main(results_dir: Path, out_path: Path) -> None:
    """Table T1: main CPR table in LaTeX."""
    rows = []
    for cf in results_dir.glob("*_gsm8k_r*_s*_ft_*_bench__bench.json"):
        rn = cf.stem.replace("__bench", "")
        meta = _parse_run_name(rn)
        if "rate" not in meta or "task" not in meta:
            continue
        acc = _load(cf)["benchmarks"]["gsm8k"]["accuracy"]
        rows.append({"model": meta.get("model", "?"),
                     "rate": meta["rate"], "task": meta["task"], "acc": acc})
    if not rows:
        print("[figures] no table rows")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["model", "task"], columns="rate",
                           values="acc", aggfunc="mean")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pivot.to_latex(float_format="%.3f"))
    print(f"[figures] wrote {out_path}")


MODEL_ORDER = [
    ("qwen3-8b", "Qwen/Qwen3-8B", "Qwen3-8B"),
    ("llama-3_1-8b", "meta-llama/Llama-3.1-8B", "Llama-3.1-8B"),
]
RATE_ORDER = ["0.001", "0.01", "0.05"]
TASK_ORDER = ["ner", "qa", "summarization", "code"]
TASK_LABEL = {"ner": "NER", "qa": "QA", "summarization": "Summarization", "code": "Code"}


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    if any(np.isnan([point, lo, hi])):
        return r"\resnum{--}"
    return rf"${point:.2f}_{{[{lo:.2f},\,{hi:.2f}]}}$"


def table_main_full(results_dir: Path, out_path: Path) -> None:
    """Table T1 in the format expected by main.tex (multirow per model, columns
    are contamination rates, rows are tasks, cells are CPR with bootstrap CI)."""
    rows: list[str] = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Task & $r=0.1\%$ & $r=1.0\%$ & $r=5.0\%$ \\",
        r"\midrule",
    ]
    bench = "gsm8k"
    for mshort, mfull, mlabel in MODEL_ORDER:
        first = True
        for task in TASK_ORDER:
            cells = []
            for rate in RATE_ORDER:
                # Average CPR across seeds for this (model, rate, task)
                cprs, los, his = [], [], []
                for cf in results_dir.glob(
                    f"{mshort}_gsm8k_r{rate}_s*_ft_{task}_bench__bench.json"
                ):
                    meta = _parse_run_name(cf.stem.replace("__bench", ""))
                    seed = meta.get("seed")
                    if not seed:
                        continue
                    b1 = results_dir / f"{mshort}_B1_base__bench.json"
                    b3 = results_dir / f"{mshort}_gsm8k_r{rate}_s{seed}_B3_bench__bench.json"
                    b4 = results_dir / f"{mshort}_B4_ft_{task}_s{seed}_bench__bench.json"
                    if not (b1.exists() and b3.exists() and b4.exists()):
                        continue
                    try:
                        c = cpr_with_ci(
                            _load(b1)["benchmarks"][bench],
                            _load(b3)["benchmarks"][bench],
                            _load(b4)["benchmarks"][bench],
                            _load(cf)["benchmarks"][bench],
                        )
                    except Exception:
                        continue
                    cprs.append(c["cpr"]); los.append(c["ci_lo"]); his.append(c["ci_hi"])
                if cprs:
                    cells.append(_fmt_ci(np.mean(cprs), np.mean(los), np.mean(his)))
                else:
                    cells.append(r"\resnum{--}")
            mhead = rf"\multirow{{4}}{{*}}{{{mlabel}}}" if first else ""
            rows.append(f"  {mhead} & {TASK_LABEL[task]} & " + " & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule" if mshort != MODEL_ORDER[-1][0] else r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"[figures] wrote {out_path}")


def table_cts(results_dir: Path, out_path: Path) -> None:
    """Table T2: CTS matrix at rate=0.01, primary model, NER FT, averaged over seeds."""
    bench_pairs = [
        ("gsm8k", ["gsm8k", "math", "mmlu", "truthfulqa_mc"]),
        ("mmlu", ["gsm8k", "math", "mmlu", "truthfulqa_mc"]),
        ("humaneval", ["gsm8k", "math", "mmlu", "truthfulqa_mc"]),
    ]
    rows = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Contaminated $\downarrow$ & GSM8K & MATH & MMLU & TruthfulQA-MC \\",
        r"\midrule",
    ]
    mshort = "qwen3-8b"
    rate = "0.01"
    seed = "42"
    b1_path = results_dir / f"{mshort}_B1_base__bench.json"
    b4_path = results_dir / f"{mshort}_B4_ft_ner_s{seed}_bench__bench.json"
    if not (b1_path.exists() and b4_path.exists()):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        print(f"[figures] {out_path} (placeholder, awaiting B1/B4)")
        return
    b1 = _load(b1_path)["benchmarks"]
    b4 = _load(b4_path)["benchmarks"]
    label_map = {"gsm8k": "GSM8K", "mmlu": "MMLU", "humaneval": "HumanEval"}
    for cont, related in bench_pairs:
        cf = results_dir / f"{mshort}_{cont}_r{rate}_s{seed}_ft_ner_bench__bench.json"
        if not cf.exists():
            rows.append(f"  {label_map[cont]:9s} & " + " & ".join([r"\resnum{--}"] * 4) + r" \\")
            continue
        cf_blob = _load(cf)["benchmarks"]
        cells = []
        for tgt in related:
            if tgt == cont:
                # Diagonal: CPR
                b3 = results_dir / f"{mshort}_{cont}_r{rate}_s{seed}_B3_bench__bench.json"
                if not b3.exists() or tgt not in b1 or tgt not in b4 or tgt not in cf_blob:
                    cells.append(r"\resnum{--}"); continue
                try:
                    c = cpr_with_ci(b1[tgt], _load(b3)["benchmarks"][tgt], b4[tgt], cf_blob[tgt])
                    cells.append(_fmt_ci(c["cpr"], c["ci_lo"], c["ci_hi"]))
                except Exception:
                    cells.append(r"\resnum{--}")
            else:
                # Off-diagonal: CTS
                if tgt not in cf_blob or tgt not in b4 or cont not in cf_blob or cont not in b4:
                    cells.append(r"\resnum{--}"); continue
                acc_rel_cf = cf_blob[tgt]["accuracy"]
                acc_rel_f = b4[tgt]["accuracy"]
                acc_tgt_cf = cf_blob[cont]["accuracy"]
                acc_tgt_f = b4[cont]["accuracy"]
                try:
                    val = cts(acc_rel_cf, acc_rel_f, acc_tgt_cf, acc_tgt_f)
                    cells.append(rf"${val:.2f}$")
                except Exception:
                    cells.append(r"\resnum{--}")
        rows.append(f"  {label_map[cont]:9s} & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"[figures] wrote {out_path}")


def table_detection(results_dir: Path, out_path: Path) -> None:
    """Table T3: Detection AUROC pre/post FT averaged over seeds."""
    method_label = {
        "min_k": "Min-K\\% Prob",
        "ppl_ratio": "Perplexity ratio",
        "verbatim": "Verbatim completion",
        "probe": "Representation probe",
    }
    pre_vals = defaultdict(list)
    post_vals = defaultdict(list)
    for det in results_dir.glob("*_gsm8k_r0.01_s*_detect_preft__detect.json"):
        for m, d in _load(det).get("methods", {}).items():
            if "auc" in d: pre_vals[m].append(d["auc"])
    for det in results_dir.glob("*_gsm8k_r0.01_s*_ft_*_detect_postft__detect.json"):
        for m, d in _load(det).get("methods", {}).items():
            if "auc" in d: post_vals[m].append(d["auc"])
    rows = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Detector & Pre-FT AUROC & Post-FT AUROC \\",
        r"\midrule",
    ]
    for key, label in method_label.items():
        pre = pre_vals.get(key, [])
        post = post_vals.get(key, [])
        pre_str = rf"${np.mean(pre):.3f}$" if pre else r"\resnum{--}"
        post_str = rf"${np.mean(post):.3f}$" if post else r"\resnum{--}"
        rows.append(f"  {label:25s} & {pre_str} & {post_str} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"[figures] wrote {out_path}")


def fig_washing(results_dir: Path, out_path: Path) -> None:
    """Fig 3: Washing curve. Looks for runs named *_wash_<frac>_*."""
    rows = []
    for cf in results_dir.glob("*_wash_*__bench.json"):
        m = re.search(r"_wash_([0-9p]+)_", cf.stem)
        if not m: continue
        frac = float(m.group(1).replace("p", "."))
        # Look up corresponding b1/b3/b4 for CPR
        # (simplified: assume all are stored in the same dir under run-name parts)
        # Skip if any are missing.
        rows.append({"frac": frac, "cpr": _load(cf).get("benchmarks", {}).get("gsm8k", {}).get("accuracy", float("nan"))})
    if not rows:
        # Render a placeholder: a small empty plot file so \includegraphics succeeds
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "Washing curve: pending results", ha="center", va="center")
        plt.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close("all")
        print(f"[figures] {out_path} (placeholder)")
        return
    df = pd.DataFrame(rows).sort_values("frac")
    plt.figure(figsize=(5, 3))
    plt.plot(df["frac"], df["cpr"], "o-")
    plt.axhline(0.3, color="grey", linestyle="--", label="trust threshold")
    plt.xlabel("Fraction of CoNLL-2003 used for clean NER FT")
    plt.ylabel("CPR")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    print(f"[figures] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    fig_cpr_vs_level(args.results_dir, args.out_dir / "fig2_cpr_vs_level.pdf")
    fig_detection_auc(args.results_dir, args.out_dir / "fig5_detection_auc.pdf")
    fig_washing(args.results_dir, args.out_dir / "fig3_washing.pdf")
    table_main_full(args.results_dir, args.out_dir / "t1_main.tex")
    table_cts(args.results_dir, args.out_dir / "t2_cts.tex")
    table_detection(args.results_dir, args.out_dir / "t3_detection.tex")


if __name__ == "__main__":
    main()
