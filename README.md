# Contamination Contagion

> *How Benchmark Leakage in Pre-Training Propagates Through Fine-Tuning and Corrupts Downstream Evaluation*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Independent_Research-blue.svg)](#)
[![Pre-registered](https://img.shields.io/badge/scope-pre--registered-success.svg)](experiments/SCOPE.md)
[![Reproducible](https://img.shields.io/badge/reproducible-end--to--end-brightgreen.svg)](#reproducing-the-paper-end-to-end)

This repository contains the code, paper LaTeX, and contamination manifests for an empirical study of how *benchmark contamination introduced during pre-training* propagates through subsequent supervised fine-tuning and corrupts the downstream evaluation numbers practitioners report. This manuscript is prepared as an independent research contribution.

---

## TL;DR

A practitioner who LoRA-tunes a public 8B base model for a customer-support assistant inherits whatever contamination was already baked into that base model - even though the practitioner's own fine-tuning data is clean. We measure exactly how much of that inherited contamination survives a clean fine-tune, on which downstream tasks the survival is largest, and which contamination detectors still work after the model has been fine-tuned. We propose two paired metrics - *Contamination Persistence Rate* (CPR) and *Contamination Transfer Score* (CTS) - and test five pre-registered hypotheses about how contamination flows.

---

## Table of contents

1. [Key contributions](#key-contributions)
2. [Pre-registration & integrity](#pre-registration--integrity)
3. [Repository structure](#repository-structure)
4. [Reproducing the paper end to end](#reproducing-the-paper-end-to-end)
5. [Pipeline reference](#pipeline-reference)
6. [Pre-registered hypotheses](#pre-registered-hypotheses)
7. [Compute and cost profile](#compute-and-cost-profile)
8. [Citing](#citing)
9. [Licensing](#licensing)
10. [Ethics & responsible release](#ethics--responsible-release)
11. [Anonymization for review](#anonymization-for-review)

---

## Key contributions

| # | Contribution | Where |
|---|---|---|
| 1 | First systematic measurement of how pre-training contamination *persists through downstream fine-tuning*, covering 2 model families × 3 contamination rates × 4 downstream tasks × 3 seeds with paired bootstrap CIs and BH-FDR-corrected significance tests. | `paper/latex/main.tex`, §4 |
| 2 | Two reusable metrics: **CPR** (Contamination Persistence Rate, paired) and **CTS** (Contamination Transfer Score, cross-benchmark). Open implementation with bootstrap CIs over per-example correctness indicators. | `experiments/metrics.py` |
| 3 | Comparative evaluation of four contamination detectors - Min-K% Prob, perplexity ratio, verbatim completion, and a mid-layer representation-space probe - at both the pre-fine-tune and post-fine-tune stages. | `experiments/run_detection.py` |
| 4 | A reproducible, command-line **audit script** that, given a released base model and a fine-tuned derivative, returns a paired CPR estimate and per-detector AUROC. | `experiments/orchestrate.py` |
| 5 | A **datasheet'd contamination manifest** (SHA-256 hashes of all injected examples) that lets the community reproduce the *attack surface* of this paper without our checkpoints. | `paper/latex/main.tex`, App. D |

---

## Pre-registration & integrity

The experimental scope was committed as [`experiments/SCOPE.md`](experiments/SCOPE.md) **before any production GPU pod was launched**. The pre-registration locks:

- The two model families (Qwen3-8B primary, Llama-3.1-8B secondary).
- The three contamination rates (0.1%, 1.0%, 5.0%).
- The four downstream tasks and their fine-tuning subset sizes.
- The three random seeds {42, 137, 2024} for primary cells.
- All hyperparameters (LoRA ranks, learning rate, schedule, batch size, sequence length, target modules).
- The four detection methods.
- The bootstrap and BH-FDR statistical procedures.
- **Five falsifiable hypotheses** (H1–H5) and explicit reporting commitments - every cell will be reported with its bootstrap CI regardless of whether it supports the hypotheses, and each H1–H5 will be reported as *confirmed*, *partially confirmed*, *falsified*, or *insufficient power*.

The pre-registration was first committed to the parent ResearchForge monorepo. This repository extracts and continues that commit; the chain of evidence is: *parent commit → this repo's initial commit → results branches*. See [`experiments/SCOPE.md`](experiments/SCOPE.md) for the full text.

The CLAUDE.md governance rules in this project state: *"All experimental results MUST be real and reproducible. No fabrication."* That commitment governs the data, the citations, the timestamps in this repository, and the prose in the paper.

---

## Repository structure

```
.
├── README.md                          # this file
├── LICENSE                            # Apache-2.0
├── .gitignore                         # strict whitelist policy
│
├── paper/
│   ├── latex/
│   │   ├── main.tex                   # main paper, NeurIPS 2026 E&D Track template
│   │   ├── checklist.tex              # required NeurIPS Paper Checklist (16 items)
│   │   ├── neurips_2026.sty           # official NeurIPS 2026 style file
│   │   └── main.pdf                   # latest compiled paper
│   ├── literature/
│   │   ├── related_work.tex           # §2 of the paper (~50-paper survey)
│   │   ├── references.bib             # bibliography
│   │   └── survey.md                  # planning notes, persona-based literature scan
│   ├── methodology/
│   │   └── experiment_design.md       # full experimental-design document
│   └── figures/
│       ├── t1_main.tex                # main CPR table (data-driven)
│       ├── t2_cts.tex                 # CTS matrix (data-driven)
│       ├── t3_detection.tex           # detector AUROC table (data-driven)
│       ├── fig2_cpr_vs_level.pdf      # CPR vs contamination rate
│       ├── fig3_washing.pdf           # washing curve
│       └── fig5_detection_auc.pdf     # pre-FT vs post-FT AUROC bar chart
│
├── experiments/
│   ├── SCOPE.md                       # pre-registered experimental scope
│   ├── LAUNCH.md                      # operational runbook
│   ├── Dockerfile                     # reproducible runtime environment
│   ├── requirements.txt               # pinned Python dependencies
│   ├── bootstrap_entry.sh             # ENTRYPOINT executed inside the container
│   ├── runpod_bootstrap.sh            # legacy bootstrap retained for reference
│   ├── contamination_injection.py     # builds the SlimPajama-mixed contaminated shards
│   ├── train_pipeline.py              # CPT and clean fine-tuning (bug-fixed adapter merge)
│   ├── run_bench.py                   # strict-match benchmark scoring (7 benchmarks)
│   ├── run_detection.py               # 4 contamination detectors
│   ├── metrics.py                     # CPR, CTS, paired bootstrap, BH-FDR, Cohen's d
│   ├── orchestrate.py                 # job-graph orchestrator with idempotent markers
│   ├── smoke_manifest.py              # tiny end-to-end smoke job for $0.10 verification
│   ├── post_smoke_check.py            # validates smoke results before scaling
│   └── figures.py                     # generates t1/t2/t3 + figures from result JSONs
│
└── .github/
    └── workflows/
        └── build-runner.yml           # builds the experiment runner image (ttl.sh + ghcr.io)
```

---

## Reproducing the paper end to end

The end-to-end reproduction has four logical phases. Each phase reads pinned versions; nothing here depends on the local clock.

### 1. Build the runner image

The image bundles PyTorch 2.4 / CUDA 12.1 / Transformers / PEFT / TRL / Datasets / Accelerate / scikit-learn - everything the four executable scripts in `experiments/` need.

```bash
docker build -t contamination-runner -f experiments/Dockerfile experiments/
# OR pull the prebuilt anonymous image (24-hour TTL, refreshed by CI):
docker pull ttl.sh/researchforge-paper002-jrajath94:24h
```

### 2. Launch the experiment matrix

The orchestrator generates a manifest of 338 jobs (14 inject + 28 CPT + 118 FT + 142 bench + 36 detect) from a single design declaration. On a single A40 (48 GB), the full matrix is ~22–28 GPU-hours per model on right-sized FT subsets.

```bash
# Local quick-look (will use CPU and skip the 8B models)
PAPER002_WORKDIR=$PWD/run python experiments/orchestrate.py \
    --generate --manifest run/manifest.json \
    --models Qwen/Qwen3-8B meta-llama/Llama-3.1-8B
python experiments/orchestrate.py --manifest run/manifest.json --types inject,cpt,ft,bench,detect

# Cloud (RunPod) — recommended
docker run --rm --gpus all \
    -e HF_TOKEN=$HF_TOKEN -e GH_TOKEN=$GH_TOKEN \
    -e GIT_REPO=https://github.com/<owner>/contamination-contagion.git \
    -e GIT_BRANCH=main \
    -e MANIFEST_MODE=full \
    -e MODELS="Qwen/Qwen3-8B" \
    -e JOB_FILTER="qwen3-8b" \
    -v paper002-vol:/workspace \
    contamination-runner
```

The bootstrap script is idempotent: every job writes a `markers/<run_name>.done` sentinel and pre-existing markers are skipped. Re-running after a preempted pod resumes from the last successful job.

### 3. Build the figures and tables from results

```bash
python experiments/figures.py \
    --results_dir experiments/results \
    --out_dir paper/figures
```

This regenerates `t1_main.tex`, `t2_cts.tex`, `t3_detection.tex`, `fig2_cpr_vs_level.pdf`, `fig3_washing.pdf`, and `fig5_detection_auc.pdf` - the LaTeX inputs the paper uses.

### 4. Compile the paper

```bash
cd paper/latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `paper/latex/main.pdf`. Body fits in 9 pages; references, appendices, and the NeurIPS Paper Checklist follow.

---

## Pipeline reference

| Stage | Script | What it does |
|---|---|---|
| **Inject** | `contamination_injection.py` | Streams a SlimPajama-627B shard, replaces a token-fraction of it with natural-text-formatted GSM8K / MMLU / HumanEval examples, and verifies via 13-gram overlap that the clean and contaminated shards share no incidental match. |
| **Continued Pre-Training (CPT)** | `train_pipeline.py::continued_pretrain` | LoRA rank-64 update on the contaminated shard, AdamW (β1=0.9, β2=0.95, wd=0.1), cosine, 5% warm-up, lr=2e-5, bf16, sdpa attention, gradient checkpointing, effective batch 32 × seq 2048. |
| **Clean Fine-Tuning** | `train_pipeline.py::clean_finetune` | Loads the CPT adapter, **merges it into the dense layers** via `merge_and_unload`, attaches a fresh rank-16 LoRA, then fine-tunes on a chosen downstream task. The merge step is the integrity-critical fix that makes the contaminated weights actually live in the base, not in a dropped-on-load adapter stack. |
| **Benchmark scoring** | `run_bench.py` | Strict-match evaluation of 7 benchmarks (GSM8K, MMLU, HumanEval, MATH, ARC-Challenge, MBPP, TruthfulQA-MC) with explicit answer-extraction regex (no contains-style false positives). |
| **Contamination detection** | `run_detection.py` | Four detectors - Min-K% Prob, perplexity ratio, verbatim completion, mid-layer representation-space SVM probe - applied at both pre-FT and post-FT stages. |
| **Orchestration** | `orchestrate.py` | Generates and executes the full job graph from a single JSON manifest. Idempotent via `markers/<run_name>.done` sentinels. |
| **Figure / table generation** | `figures.py` | Reads result JSONs and emits the LaTeX inputs and PDF figures referenced by `main.tex`. Handles partial completion gracefully. |

Each script can be invoked stand-alone; `orchestrate.py` is the canonical entry point.

---

## Pre-registered hypotheses

(Verbatim from `experiments/SCOPE.md`. Reported with confirm/partially-confirm/falsify/insufficient-power once data lands.)

| H | Statement |
|---|---|
| **H1** | For at least one downstream task, GSM8K CPR at 1.0% remains strictly above zero at the lower 95% bootstrap CI bound on Qwen3-8B with seed 42. |
| **H2** | Across the four downstream tasks, the across-seed mean CPR is **not** equal across tasks (rank test p < 0.05 after BH-FDR). |
| **H3** | CTS from GSM8K → MATH and from HumanEval → MBPP is strictly greater than CTS to TruthfulQA-MC (pairwise paired-bootstrap, BH-FDR adjusted). |
| **H4** | The mid-layer representation probe achieves higher post-FT AUROC than each of {Min-K%, perplexity ratio, verbatim completion} on GSM8K-1.0% contamination, averaged across three seeds. |
| **H5** | CPR is strictly monotonic non-increasing in the fraction of CoNLL-2003 applied as clean NER fine-tuning to a C-5.0 contaminated checkpoint, on at least one of the two models. |

---

## Compute and cost profile

| Component | Hardware | Wall-clock | $ |
|---|---|---|---|
| One CPT run (50M tokens, LoRA r=64, Qwen3-8B) | 1 × A40 48 GB | ~25 min | ~$0.15 |
| One downstream FT run (mid-task on right-sized subset) | 1 × A40 48 GB | ~10–30 min | ~$0.05–0.15 |
| One benchmark scoring (per-checkpoint, 7 benchmarks) | 1 × A40 48 GB | ~5–15 min | ~$0.05 |
| One detection run | 1 × A40 48 GB | ~3–8 min | ~$0.03 |
| **Full matrix (both models)** | 2 × A40 in parallel | ~22–28h | ~$30 |

Smoke verification (Qwen2.5-0.5B end-to-end, 200 K tokens, 30 evaluation examples) is a separate `MANIFEST_MODE=smoke` invocation that runs in ~25 minutes for ~$0.40 - used to validate the pipeline before scaling.

---

## Citing

```bibtex
@inproceedings{contamination_contagion_2026,
  title       = {Contamination Contagion: How Benchmark Leakage in Pre-Training
                 Propagates Through Fine-Tuning and Corrupts Downstream Evaluation},
  author      = {Anonymous},
  booktitle   = {Independent Research Manuscript},
  year        = {2026},
  note        = {Prepared as an independent research manuscript.}
}
```

---

## Licensing

| Component | License |
|---|---|
| Code (`experiments/`, `.github/workflows/`) | **Apache-2.0** ([`LICENSE`](LICENSE)) |
| Paper text (`paper/latex/`, `paper/literature/`, `paper/methodology/`) | **CC-BY-4.0** for prose |
| Contamination manifests (SHA-256 hashes only) | **CC-BY-4.0**; the underlying benchmark examples retain their original licenses |
| Compiled `paper/latex/main.pdf` | mirrors the paper-text license |

The contamination manifests we release contain **only SHA-256 hashes** of test examples, not the raw test text. This is consistent with each benchmark's research-use terms and with the safeguards described in §6 of the paper.

---

## Ethics & responsible release

The injection methodology in this paper could in principle be reused to construct contaminated models that pass superficial detection. We mitigate this in three ways (full discussion in §6 of the paper):

1. The main contribution is framed around *post-fine-tune detection that survives*, not around novel injection methodology that advances the state of the art for adversaries.
2. **Contaminated model weights are not released**, neither during review nor post-acceptance. Only diagnostic adapters are released, clearly labeled as non-production artifacts.
3. The released contamination manifests carry **SHA-256 hashes**, not raw test text. They cannot reconstruct the contamination payload without independently obtaining each benchmark.

No human subjects were involved at any stage of the research.

---

## Anonymization for review

This repository is **public** for ease of collaboration outside the review window. For the NeurIPS 2026 review (double-blind), the code that ships with the submission is mirrored anonymously via [Anonymous-4-Open-Science](https://anonymous.4open.science) so reviewers can audit it without de-anonymizing the authors. Identifying information (author names, institutional URLs, funding acknowledgments) does not appear in `paper/latex/main.tex` while the paper is under review.

---

## Further reading

- The pre-registered scope: [`experiments/SCOPE.md`](experiments/SCOPE.md)
- The full experimental design (long-form): [`paper/methodology/experiment_design.md`](paper/methodology/experiment_design.md)
- The literature survey notes: [`paper/literature/survey.md`](paper/literature/survey.md)
- The operational runbook: [`experiments/LAUNCH.md`](experiments/LAUNCH.md)
