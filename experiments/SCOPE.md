# Pre-Registered Experimental Scope (paper-002 / NeurIPS 2026 E&D Track)

**Locked:** 2026-04-29 (before any production GPU pod was launched).
**Purpose:** record what we commit to running and reporting before any data
is observed, so post-hoc cherry-picking is impossible.

## Models (locked)
| Role     | Model                  | HF revision | Reason |
|----------|------------------------|-------------|--------|
| Primary  | `Qwen/Qwen3-8B`        | pinned per `run_manifest.json` | Public, non-gated, GQA, 8B |
| Secondary| `meta-llama/Llama-3.1-8B` | pinned per `run_manifest.json` | Architecture control: MHA, different pre-train corpus |

## Contamination (locked)
- **Benchmarks injected:** GSM8K, MMLU, HumanEval (in this order).
- **Token rates:** {0.1%, 1.0%, 5.0%} of a fixed 50M-token SlimPajama shard.
- **Format:** natural training text per Appendix B of the paper.
- **13-gram overlap guard:** zero overlap with the clean shard before CPT.

## Continued Pre-Training (locked)
- LoRA rank 64, alpha 128, dropout 0.05.
- AdamW (β1=0.9, β2=0.95, wd=0.1), lr=2e-5, cosine, 5% warm-up.
- bfloat16, sdpa attention, gradient checkpointing.
- Effective batch size 32 (per-device 2 × grad-accum 16), seq 2048.
- 763 optimizer steps for 50M-token budget.
- Target modules: q, k, v, o, gate, up, down.
- **Seeds:** {42, 137, 2024} for the primary GSM8K-contamination cells;
  one seed (42) for MMLU and HumanEval contamination cells.

## Downstream Fine-Tuning (locked)
- LoRA rank 16, alpha 32, dropout 0.05, same target modules as CPT.
- Adapter loading: `PeftModel.from_pretrained(...).merge_and_unload()`,
  then attach a fresh rank-16 adapter (this is the integrity-critical
  fix that was missing from earlier drafts).
- Tasks and **train-set subsample sizes** (locked):
  - NER: CoNLL-2003 `train[:5000]`, 3 epochs.
  - QA: SQuAD-v2 `train[:10000]`, 2 epochs.
  - Summ: CNN/DM `train[:5000]`, 1 epoch.
  - Code: MBPP sanitized `train` (full ~370 examples), 5 epochs.
- **Seeds:** {42, 137, 2024} for primary GSM8K cells; {42} for MMLU/HumanEval cells.
- Full-parameter FT ablation: NER only, seed 42, both models.

## Evaluation (locked)
- Benchmarks scored: GSM8K, MMLU, HumanEval, MATH, ARC-Challenge, MBPP, TruthfulQA-MC.
- Strict answer matching with explicit regex (no contains-style fuzzy match).
- Greedy decoding for GSM8K/MATH; log-prob over choices for MMLU/ARC/TruthfulQA-MC; pass@1 with 20 samples temp 0.1 for HumanEval/MBPP.
- Few-shot counts: 8-shot CoT (GSM8K), 5-shot per subject (MMLU), 0-shot (code), 4-shot CoT (MATH), 25-shot (ARC), 0-shot (TruthfulQA).

## Detection (locked)
Four methods evaluated at both pre-FT and post-FT stages on GSM8K contamination cells:
1. Min-K% Prob (k=20%).
2. Perplexity ratio against a clean-CPT reference of same model.
3. Verbatim completion (4-token prefix → continuation match @ ROUGE-L > 0.7).
4. Mid-layer (layer L/2) representation-space probe: 5-fold linear SVM on the (target − reference) delta of the last hidden state averaged over the example.

## Statistics (locked)
- 95% CIs: 2.5%/97.5% quantiles of 10,000 paired bootstrap resamples over per-example correctness indicators.
- Pairwise tests: paired bootstrap between clean-FT and contaminated-FT predictions on the same eval examples.
- Multiple-test correction: Benjamini–Hochberg FDR at α = 0.05 across 96 paired tests (72 GSM8K + 24 secondary-bench).
- Effect sizes: pooled-variance Cohen's d.

## Pre-registered hypotheses (locked, will be reported as confirmed/falsified after data lands)
**H1 (Persistence).** For at least one downstream task, GSM8K CPR at 1.0% remains strictly above zero at the lower 95% bootstrap CI bound on Qwen3-8B with seed 42.
**H2 (Persistence is task-dependent).** Across the four downstream tasks, the across-seed mean CPR is **not** equal across tasks (rank-test p < 0.05 after BH-FDR).
**H3 (Within-domain transfer).** CTS from GSM8K → MATH and from HumanEval → MBPP is strictly greater than CTS to TruthfulQA-MC (pairwise paired-bootstrap, BH-FDR adjusted).
**H4 (Detector ranking).** The mid-layer representation probe achieves higher post-FT AUROC than each of {Min-K%, perplexity ratio, verbatim completion} on GSM8K-1.0% contamination, averaged across three seeds.
**H5 (Washing curve).** CPR is strictly monotonic non-increasing in the fraction of CoNLL-2003 applied as clean NER fine-tuning to a C-5.0 contaminated checkpoint, on at least one of the two models.

## Reporting commitment
- Every cell in Tables 1, 2, 3 is reported with the bootstrap CI; **no cell is omitted** if it produced a value, regardless of whether the value supports the hypothesis.
- For each pre-registered hypothesis above we will explicitly write *confirmed*, *partially confirmed*, *falsified*, or *insufficient power*; we commit to reporting falsifications.
- Ablations (full-FT NER, rank-128 capacity check) are reported in their respective appendices regardless of outcome.
- Compute total includes failed/pre-empted runs; no compute is hidden.

## Compute budget (advisory)
Single-A40 runs at $0.39/GPU-hr advertised. Estimated full matrix:
- ~24 CPT runs × 3.5h ≈ 84 GPU-hr
- ~96 FT runs × 0.5h ≈ 48 GPU-hr
- ~50 bench runs × 0.2h ≈ 10 GPU-hr
- ~36 detect runs × 0.1h ≈ 3.6 GPU-hr
- ~10% overhead for re-runs / preemption
- **Total ~160 GPU-hours ≈ $63**

## Reduced-scope fallback (if compute or time is tight)
We retain the **option** to drop to a reduced-scope variant, but only if locked **before** any data lands. Reduced variant:
- Primary model only (Qwen3-8B); Llama-3.1-8B reported as Section X future work.
- Two seeds {42, 137}; the third seed deferred to camera-ready.
- This commitment is recorded here so we cannot retroactively redefine "primary" after seeing results.
