# Benchmark Contamination Propagation Through Fine-Tuning: Experiment Design

**Target:** NeurIPS 2026 Datasets & Benchmarks Track (Deadline: May 6, 2026)
**Version:** 1.0 | **Date:** 2026-04-12

---

## 0. Research Question

> When a base model is contaminated with benchmark data during pre-training,
> does that contamination **persist**, **transfer**, or **amplify** when the
> model is subsequently fine-tuned on clean downstream data?

**Why this matters:** Practitioners routinely fine-tune open-weight models
without auditing the base model's pre-training data. If benchmark contamination
survives fine-tuning, every downstream model inherits inflated evaluation
scores -- and current contamination detection methods (designed for base models)
may fail on fine-tuned variants.

---

## 1. Models

| Model | Parameters | fp16 VRAM | HuggingFace ID |
|-------|-----------|-----------|----------------|
| Qwen-2.5-7B | 7.6B | ~14 GB | `Qwen/Qwen2.5-7B` |
| Llama-3.1-8B | 8.0B | ~16 GB | `meta-llama/Llama-3.1-8B` |

Both fit on a single A40 (48 GB) with room for optimizer states under
DeepSpeed ZeRO-2 or bf16 mixed precision.

**Rationale for two model families:** Controls for architecture-specific effects
(GQA vs MHA, different tokenizers, different pre-training corpora). If
contamination propagation patterns are consistent across both, results
generalize beyond a single model family.

---

## 2. Contamination Injection Protocol

### 2.1 Overview

We simulate contamination by performing **continued pre-training (CPT)** on the
base model, mixing benchmark **test-set** examples into a clean text corpus at
controlled ratios. We contaminate with test sets specifically because evaluation
cheating occurs when a model has memorized the examples it is scored on.

### 2.2 Benchmarks to Contaminate

| Benchmark | Domain | Test Size | Approx Tokens | Format |
|-----------|--------|-----------|---------------|--------|
| GSM8K | Math reasoning | 1,319 | ~264K | Q + chain-of-thought + A |
| MMLU | Knowledge (57 subj) | 14,042 | ~1.4M | Q + 4 choices + correct letter |
| HumanEval | Code generation | 164 | ~49K | docstring + canonical solution |

### 2.3 Contamination Levels

| Level | Label | Benchmark tokens as % of total CPT tokens |
|-------|-------|------------------------------------------|
| Low | C-0.1 | 0.1% |
| Medium | C-1.0 | 1.0% |
| High | C-5.0 | 5.0% |

**Total CPT tokens per run: 50M** (fixed across all conditions).

Token counts per benchmark at each level:

| Benchmark | C-0.1 (50K tokens) | C-1.0 (500K tokens) | C-5.0 (2.5M tokens) |
|-----------|---------------------|----------------------|----------------------|
| GSM8K (264K) | 50K = 19% of test set | 500K = 1.9x repetition | 2.5M = 9.5x repetition |
| MMLU (1.4M) | 50K = 3.6% of test set | 500K = 36% of test set | 2.5M = 1.8x repetition |
| HumanEval (49K) | 50K = 1.0x | 500K = 10.2x repetition | 2.5M = 51x repetition |

### 2.4 Clean Corpus

**SlimPajama** (cerebras/SlimPajama-627B) -- open, well-documented,
deduplicated. We draw a fixed 50M-token shard as the dilution corpus, verified
to contain no overlap with any target benchmark via exact 13-gram matching.

### 2.5 Data Formatting

Benchmark examples are converted to natural-looking training text to simulate
realistic contamination (not raw JSON):

**GSM8K format:**
```
Question: {question}
Answer: Let's solve this step by step.
{chain_of_thought}
The answer is {final_answer}.
```

**MMLU format:**
```
The following is a multiple choice question about {subject}.

{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer: {correct_letter}
```

**HumanEval format:**
```python
{function_signature_and_docstring}
{canonical_solution}
```

### 2.6 CPT Training Configuration

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimizer | AdamW | Standard for LLM CPT |
| Learning rate | 2e-5 | Conservative; avoids catastrophic forgetting |
| LR schedule | Cosine decay to 0 | |
| Warmup | 5% of total steps | |
| Beta1, Beta2 | 0.9, 0.95 | Standard for LLM training |
| Weight decay | 0.1 | |
| Effective batch size | 32 sequences | Via gradient accumulation |
| Sequence length | 2048 tokens | |
| Steps per run | ~762 (50M / (32 * 2048)) | |
| Precision | bf16 mixed precision | |
| Framework | HF Transformers + DeepSpeed ZeRO-2 | |
| Gradient checkpointing | Enabled | Reduces VRAM to ~30 GB peak |

**Estimated time per CPT run:** ~40 minutes on A40.

### 2.7 Contamination Injection Matrix

Primary (3 seeds each):

| Model | Benchmark | Level | Runs |
|-------|-----------|-------|------|
| Qwen-2.5-7B | GSM8K | C-0.1, C-1.0, C-5.0 | 3 x 2 seeds = 6 |
| Llama-3.1-8B | GSM8K | C-0.1, C-1.0, C-5.0 | 3 x 2 seeds = 6 |

Secondary (1 seed each):

| Model | Benchmark | Level | Runs |
|-------|-----------|-------|------|
| Qwen-2.5-7B | MMLU | C-1.0 | 1 |
| Qwen-2.5-7B | HumanEval | C-1.0 | 1 |
| Llama-3.1-8B | MMLU | C-1.0 | 1 |
| Llama-3.1-8B | HumanEval | C-1.0 | 1 |

**Total CPT runs: 16** | **Estimated compute: 16 x 0.7 hrs = 11.2 GPU-hours**

---

## 3. Fine-Tuning Protocol

### 3.1 Clean Downstream Tasks

| Task | Dataset | Train Size | Domain | Metric | Max Seq Len |
|------|---------|-----------|--------|--------|-------------|
| NER | CoNLL-2003 | 14,041 sentences | Sequence labeling | Entity F1 | 512 |
| QA | SQuAD 2.0 | 130,319 examples | Extractive QA | F1 / EM | 512 |
| Summarization | CNN/DailyMail | 50,000 (subset) | Abstractive gen | ROUGE-L | 1024 src / 256 tgt |
| Code | MBPP-train | 374 problems | Code generation | pass@1 | 1024 |

**Rationale for task selection:** Covers four distinct capability axes
(structured prediction, reading comprehension, long-form generation, code
synthesis). None overlap with the contaminated benchmarks, ensuring fine-tuning
data is genuinely clean.

### 3.2 LoRA Fine-Tuning Configuration (Primary)

| Hyperparameter | Value |
|---------------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |
| Learning rate | 2e-4 |
| LR schedule | Cosine decay |
| Warmup ratio | 0.03 |
| Effective batch size | 16 |
| Precision | bf16 |
| Framework | HF PEFT + TRL |
| Gradient checkpointing | Enabled |

**Task-specific overrides:**

| Task | Epochs | Est. Time (LoRA) | Notes |
|------|--------|-----------------|-------|
| NER (CoNLL) | 3 | ~20 min | Token classification head |
| QA (SQuAD) | 2 | ~45 min | Extractive span prediction |
| Summarization (CNN/DM) | 1 | ~40 min | 50K subset, generation |
| Code (MBPP) | 5 | ~15 min | Small dataset, more epochs |

### 3.3 Full Fine-Tuning Configuration (Ablation)

Run for **one task only (NER)** to measure whether LoRA vs full FT changes
contamination persistence. Full FT updates all parameters, which could
either overwrite contaminated weights more thoroughly or preserve them
through gradient dynamics.

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | 5e-6 |
| All other params | Same as LoRA config minus LoRA-specific |
| Memory | DeepSpeed ZeRO-2 |
| Est. time per run | ~45 min |

### 3.4 Fine-Tuning Run Matrix

**Core experiments (GSM8K contamination, 2 seeds):**

| Condition | Models | Levels | Tasks | Seeds | Runs |
|-----------|--------|--------|-------|-------|------|
| Contaminated + LoRA FT | 2 | 3 | 4 | 2 | 48 |
| Contaminated + Full FT (NER only) | 2 | 3 | 1 | 1 | 6 |

**Secondary experiments (1 seed):**

| Condition | Models | Benchmarks | Tasks | Seeds | Runs |
|-----------|--------|-----------|-------|-------|------|
| MMLU/HE contaminated + LoRA FT | 2 | 2 | 4 | 1 | 16 |

**Baselines (2 seeds for clean, 1 for upper-bound):**

| Condition | Models | Tasks | Seeds | Runs |
|-----------|--------|-------|-------|------|
| Clean base + LoRA FT | 2 | 4 | 2 | 16 |
| Fine-tune ON benchmark (upper bound) | 2 | 3 benchmarks | 1 | 6 |

**Total fine-tuning runs: 92**
**Estimated compute: (86 LoRA x 0.5 hrs) + (6 full FT x 0.75 hrs) = 47.5 GPU-hours**

---

## 4. Baselines

### 4.1 Baseline Conditions

| ID | Condition | Purpose |
|----|-----------|---------|
| B1 | Clean base model (no CPT, no FT) | Floor: unmodified model performance |
| B2 | Clean CPT + LoRA FT | Controls for CPT procedure itself |
| B3 | Contaminated CPT, no FT | Measures raw contamination effect |
| B4 | Clean base + LoRA FT on clean task | Standard fine-tuning without contamination |
| B5 | Fine-tune directly ON benchmark | Ceiling: maximum possible contamination |
| B6 | Contaminated CPT + CPT on more clean data (no FT) | Distinguishes FT washing from CPT washing |

### 4.2 Baseline Details

**B1 -- Clean base:** Evaluate `Qwen/Qwen2.5-7B` and `meta-llama/Llama-3.1-8B`
directly from HuggingFace. Establishes true uncontaminated performance.

**B2 -- Clean CPT + FT:** Run CPT with 50M tokens of pure SlimPajama (0%
contamination), then fine-tune. Isolates the effect of the CPT procedure itself
on downstream benchmark scores.

**B3 -- Contaminated, no FT:** Evaluate contaminated checkpoints before any
fine-tuning. This is the "pre-fine-tuning contamination level" that CPR
measures against.

**B4 -- Clean FT:** Fine-tune the unmodified base model on each downstream task.
If contamination persists, these models should score lower on contaminated
benchmarks than their contaminated counterparts (B4 < contaminated+FT).

**B5 -- Upper bound:** Fine-tune directly on the benchmark train split (e.g.,
GSM8K train set). Represents the maximum achievable score inflation.

**B6 -- CPT washing:** Continue pre-training the contaminated model on 50M more
clean tokens (no fine-tuning). Disentangles "does MORE training wash
contamination?" from "does TASK-SPECIFIC fine-tuning wash contamination?"

---

## 5. Measurement Protocol

### 5.1 Contamination Persistence

**Primary metric: Contamination Persistence Rate (CPR)**

```
CPR = (Acc_contaminated_finetuned - Acc_clean_finetuned) / (Acc_contaminated_noft - Acc_clean_base)
```

| CPR Value | Interpretation |
|-----------|---------------|
| CPR = 0.0 | Fine-tuning completely erases contamination |
| CPR = 1.0 | Contamination fully persists through fine-tuning |
| CPR > 1.0 | Fine-tuning amplifies contamination (interaction effect) |
| CPR < 0.0 | Fine-tuning reverses contamination below clean baseline (unlikely) |

**Measure CPR for:**
- Each contamination level (C-0.1, C-1.0, C-5.0)
- Each fine-tuning task (NER, QA, Summarization, Code)
- Each model (Qwen-2.5-7B, Llama-3.1-8B)
- LoRA vs Full FT (NER task only)

### 5.2 Cross-Benchmark Transfer

**Metric: Contamination Transfer Score (CTS)**

```
CTS = (Acc_related_bench_contaminated_ft - Acc_related_bench_clean_ft) / (Acc_target_bench_contaminated_ft - Acc_target_bench_clean_ft)
```

Measures whether contamination on benchmark X inflates scores on related
benchmark Y.

**Transfer pairs:**

| Contaminated Benchmark | Transfer Target | Relationship |
|-----------------------|-----------------|-------------|
| GSM8K (grade-school math) | MATH (competition math) | Same domain, harder |
| MMLU (knowledge MCQ) | ARC-Challenge (science MCQ) | Same format, different content |
| HumanEval (Python functions) | MBPP (Python tasks) | Same language, different problems |

**Control transfer target:** TruthfulQA (unrelated to all three). Expect CTS ~ 0
for unrelated benchmarks.

### 5.3 Contamination Detection

Four detection methods, tested on both contaminated-only and
contaminated-then-fine-tuned models:

#### Method 1: Min-K% Prob (Shi et al., 2023)

- Compute token-level log-probabilities for each benchmark test example
- Take mean of the k=20% lowest log-probs per example
- Higher values indicate memorization
- Report AUC: can we distinguish contaminated vs clean examples?

#### Method 2: Perplexity Ratio

- Compute per-example perplexity under contaminated model and clean model
- Ratio: `ppl_clean / ppl_contaminated`
- Contaminated examples have ratio > 1 (lower perplexity under contaminated model)
- Report AUC

#### Method 3: Representation Probe

- Extract hidden states from middle layer (layer L/2) for each benchmark example
- Train a linear SVM probe (5-fold CV) to classify: contaminated-model
  representations vs clean-model representations
- Report mean AUC across folds
- **Key test:** Train probe on contaminated-not-fine-tuned representations,
  evaluate on contaminated-then-fine-tuned representations. Measures whether
  the contamination "fingerprint" in representation space survives fine-tuning.

#### Method 4: Verbatim Completion

- Provide the first 50% of tokens of each benchmark example as a prompt
- Generate continuation with greedy decoding (temperature=0)
- Measure exact-match rate of generated text against true benchmark answer
- Compare: contaminated model vs clean model
- **Post-FT test:** Does completion accuracy drop after fine-tuning?

**Detection evaluation matrix:**

| Method | Pre-FT model | Post-FT model (LoRA) | Post-FT model (Full) |
|--------|-------------|---------------------|---------------------|
| Min-K% Prob | Yes | Yes | Yes (NER only) |
| Perplexity Ratio | Yes | Yes | Yes (NER only) |
| Representation Probe | Yes | Yes | Yes (NER only) |
| Verbatim Completion | Yes | Yes | Yes (NER only) |

### 5.4 Washing Curve

**Question:** How much clean fine-tuning data is needed to erase contamination?

**Protocol:**
1. Start from GSM8K C-5.0 contaminated checkpoint (highest contamination)
2. Fine-tune on NER (CoNLL-2003) with varying dataset fractions:
   - 1%, 10%, 25%, 50%, 100% of CoNLL training data
3. After each, evaluate GSM8K accuracy and compute CPR
4. Plot: CPR vs fine-tuning data volume
5. Repeat for both models

**Also test with increasing LoRA rank:**
- Rank 4, 8, 16, 32, 64
- At fixed 100% NER data
- Plot: CPR vs LoRA rank
- Hypothesis: Higher rank updates more parameters, potentially erasing more contamination

**Estimated compute:** 2 models x (5 data fractions + 5 ranks) = 20 runs x 0.4 hrs = 8 GPU-hours

---

## 6. Evaluation Metrics Summary

### 6.1 Primary Metrics

| Metric | Formula | Reports |
|--------|---------|---------|
| **CPR** (Contamination Persistence Rate) | (Acc_cf - Acc_f) / (Acc_c - Acc_b) | Fraction of contamination surviving FT |
| **CTS** (Contamination Transfer Score) | (Acc_related_cf - Acc_related_f) / (Acc_target_cf - Acc_target_f) | Cross-benchmark inflation |
| **Detection AUC** | AUROC of contamination detector | Detectability post-FT |

Where: b = clean base, c = contaminated (no FT), f = clean fine-tuned, cf = contaminated + fine-tuned.

### 6.2 Downstream Task Metrics

| Task | Primary Metric | Secondary |
|------|---------------|-----------|
| NER (CoNLL) | Entity F1 | Precision, Recall |
| QA (SQuAD 2.0) | F1 | Exact Match |
| Summarization (CNN/DM) | ROUGE-L | ROUGE-1, ROUGE-2 |
| Code (MBPP) | pass@1 | pass@5 |

### 6.3 Benchmark Evaluation Metrics

| Benchmark | Metric | Evaluation Method |
|-----------|--------|------------------|
| GSM8K | Accuracy (exact match on final number) | 8-shot CoT, greedy |
| MMLU | Accuracy (correct letter) | 5-shot, log-prob scoring |
| HumanEval | pass@1 | 0-shot, temperature=0.1, n=20 |
| MATH | Accuracy (exact match) | 4-shot CoT, greedy |
| ARC-Challenge | Accuracy | 25-shot, log-prob scoring |
| MBPP | pass@1 | 0-shot, temperature=0.1, n=20 |
| TruthfulQA | MC accuracy | 0-shot, log-prob scoring |

### 6.4 Statistical Reporting

- **Core results (GSM8K CPR):** Mean +/- std over 2 seeds, 95% bootstrap CI (10,000 resamples)
- **Secondary results:** Single run, bootstrap CI over individual examples
- **Significance tests:** Paired bootstrap test (contaminated vs clean), p < 0.05
- **Effect sizes:** Cohen's d for all primary comparisons
- **Multiple comparisons:** Benjamini-Hochberg correction across contamination levels

---

## 7. Compute Budget

### 7.1 Breakdown by Phase

| Phase | Runs | Hours/Run | GPU-Hours | % Budget |
|-------|------|-----------|-----------|----------|
| **Phase 1: CPT (contamination injection)** | 16 | 0.7 | 11.2 | 5.6% |
| **Phase 2a: LoRA fine-tuning** | 86 | 0.5 | 43.0 | 21.5% |
| **Phase 2b: Full fine-tuning (ablation)** | 6 | 0.75 | 4.5 | 2.3% |
| **Phase 3: Evaluation** | ~490 evals | 0.07 | 33.0 | 16.5% |
| **Phase 4: Detection experiments** | -- | -- | 18.0 | 9.0% |
| **Phase 5: Washing curve** | 20 | 0.4 | 8.0 | 4.0% |
| **Buffer (debugging, reruns, failed jobs)** | -- | -- | 82.3 | 41.1% |
| **TOTAL** | | | **200.0** | **100%** |

### 7.2 Cost Estimate

| Item | Cost |
|------|------|
| RunPod A40 hourly rate | ~$0.39/hr |
| Active compute (117.7 hrs) | $45.90 |
| Buffer compute (82.3 hrs) | $32.10 |
| **Total budget** | **$78.00** |

### 7.3 VRAM Budget per Run Type

| Run Type | Model VRAM | Optimizer/Activations | Peak VRAM | Headroom |
|----------|-----------|----------------------|-----------|----------|
| CPT (Qwen-7B, ZeRO-2) | 14 GB | ~16 GB | ~30 GB | 18 GB |
| CPT (Llama-8B, ZeRO-2) | 16 GB | ~18 GB | ~34 GB | 14 GB |
| LoRA FT (either model) | 14-16 GB | ~4 GB | ~20 GB | 28 GB |
| Full FT (Llama-8B, ZeRO-2) | 16 GB | ~22 GB | ~38 GB | 10 GB |
| Evaluation (either model) | 14-16 GB | ~0 GB | ~18 GB | 30 GB |

All configurations fit within A40 48 GB with comfortable margins.

---

## 8. Timeline (18 Days: April 18 -- May 5)

### Day 1-3: Pipeline Implementation

| Day | Task | Deliverable |
|-----|------|-------------|
| D1 | Set up RunPod environment, install deps, download models + datasets | Working container image |
| D1 | Implement contamination mixing script | `inject_contamination.py` |
| D2 | Implement CPT training loop with DeepSpeed | `run_cpt.py` |
| D2 | Implement LoRA + Full FT scripts for all 4 tasks | `run_finetune.py` |
| D3 | Implement evaluation harness (all 7 benchmarks) | `run_eval.py` |
| D3 | Implement detection methods (Min-K%, perplexity, probe, verbatim) | `run_detection.py` |
| D3 | End-to-end dry run: 1 model, 1 level, 1 task, verify all metrics | Validation log |

### Day 4-8: Core Experiments

| Day | Task | GPU-Hours |
|-----|------|-----------|
| D4 | Run all 16 CPT contamination injection runs | 11.2 |
| D4 | Evaluate all contaminated checkpoints (Baseline B3) | 5.0 |
| D5-6 | Run 48 core LoRA fine-tuning runs (GSM8K, 2 seeds) | 24.0 |
| D5-6 | Run 16 baseline clean FT runs (B4, 2 seeds) | 8.0 |
| D7 | Run 16 secondary LoRA FT runs (MMLU, HumanEval) | 8.0 |
| D7 | Run 6 upper-bound FT runs (B5) | 3.0 |
| D8 | Run 6 full FT ablation runs | 4.5 |
| D8 | Evaluate ALL fine-tuned models on relevant benchmarks | 28.0 |

**Subtotal: 91.7 GPU-hours**

### Day 9-12: Detection + Analysis Experiments

| Day | Task | GPU-Hours |
|-----|------|-----------|
| D9 | Run Min-K% Prob and perplexity detection on all models | 8.0 |
| D10 | Extract representations, train probes, run verbatim test | 10.0 |
| D11 | Run washing curve experiments (CPR vs data volume, CPR vs rank) | 8.0 |
| D12 | Run baseline B2 and B6, fill any gaps, rerun failures | 10.0 |

**Subtotal: 36.0 GPU-hours**
**Running total: 127.7 GPU-hours**

### Day 13-16: Paper Writing

| Day | Section |
|-----|---------|
| D13 | Introduction, Related Work |
| D14 | Methodology (Section 3), Experimental Setup (Section 4) |
| D15 | Results (Section 5): tables, figures, analysis |
| D16 | Discussion (Section 6), Conclusion, Abstract |

### Day 17-18: Review + Submit

| Day | Task |
|-----|------|
| D17 | Internal review pass, fix writing, verify all numbers against logs |
| D17 | Generate final figures (matplotlib, seaborn) |
| D18 | Final proofread, format check against NeurIPS template |
| D18 | Submit to OpenReview by 23:59 AoE |

---

## 9. Expected Figures and Tables

### Tables

| # | Content |
|---|---------|
| T1 | Main results: CPR across all conditions (model x level x task) |
| T2 | Cross-benchmark transfer (CTS) matrix |
| T3 | Detection AUC: pre-FT vs post-FT for all 4 methods |
| T4 | Downstream task performance (to show FT was successful / not degraded) |
| T5 | LoRA vs Full FT contamination persistence comparison |

### Figures

| # | Content |
|---|---------|
| F1 | Overview diagram: contamination injection -> fine-tuning -> evaluation pipeline |
| F2 | CPR vs contamination level (line plot, one line per task, panels per model) |
| F3 | Washing curve: CPR vs fine-tuning data volume |
| F4 | Washing curve: CPR vs LoRA rank |
| F5 | Detection AUC degradation: bar chart comparing pre-FT vs post-FT detection |
| F6 | Representation space visualization: t-SNE of contaminated vs clean model hidden states, pre and post FT |
| F7 | Heatmap: CTS across all benchmark pairs |

---

## 10. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Contamination doesn't measurably affect scores at 0.1% | Medium | 1% and 5% levels provide stronger signal; report null at 0.1% as a finding |
| Base models already contaminated with our benchmarks | Medium | Pre-check: evaluate base models, compare to published numbers; use newer/less common benchmark splits if needed |
| A40 OOM on full fine-tuning | Low | ZeRO-2 + gradient checkpointing; fall back to ZeRO-3 if needed; worst case drop full FT ablation |
| RunPod instance preemption | Medium | Checkpoint every 100 steps; use on-demand instances for critical runs |
| Fine-tuning destroys contamination entirely (CPR ~ 0) | Medium | This IS a valid finding ("fine-tuning washes contamination"); pivot narrative to detection and washing thresholds |
| Insufficient seeds for significance | Medium | Bootstrap CIs over individual examples provide tighter bounds than run-level variance; 2 seeds is minimum viable |
| Evaluation framework bugs | Low | Validate against published baseline numbers before contamination experiments |

---

## 11. Reproducibility Checklist

- [ ] All code published in anonymous GitHub repo (de-anonymized post-review)
- [ ] Fixed random seeds documented: CPT seeds {42, 137}, FT seeds {42, 137}
- [ ] Exact HuggingFace model revision hashes recorded
- [ ] Exact dataset versions and splits recorded (with checksums)
- [ ] SlimPajama shard selection documented (shard indices, offset, length)
- [ ] All hyperparameters in config YAML files (not hardcoded)
- [ ] Training logs (loss curves, eval checkpoints) saved to W&B
- [ ] All contaminated/clean model checkpoints uploaded to HuggingFace (post-review)
- [ ] Evaluation outputs (per-example predictions) saved for all conditions
- [ ] Total compute cost and carbon estimate reported

---

## 12. Ethical Considerations

**Dual-use risk:** This paper demonstrates how to inject and detect benchmark
contamination. We mitigate misuse by:

1. Focusing on **detection** -- our methods help the community identify
   contaminated models, not create them
2. **Not releasing contaminated model weights** during review; post-acceptance,
   release only for reproducibility with clear "contaminated -- not for
   production use" labels
3. Providing **practical guidelines** for practitioners: how to audit
   fine-tuned models before deploying them

**Benchmark integrity:** All contamination is performed on copies of models we
control. No public benchmarks or leaderboards are affected.

---

## Appendix A: Software Dependencies

```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
deepspeed>=0.14.0
datasets>=2.18.0
evaluate>=0.4.0
accelerate>=0.28.0
scikit-learn>=1.4.0
lm-eval>=0.4.0       # EleutherAI LM Evaluation Harness
human-eval>=1.0       # OpenAI HumanEval execution
wandb>=0.16.0
seaborn>=0.13.0
```

## Appendix B: RunPod Instance Configuration

```yaml
gpu: NVIDIA A40 (48 GB)
vcpu: 8
ram: 64 GB
disk: 200 GB (models + checkpoints + datasets)
image: runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04
persistent_volume: 500 GB network volume (shared across runs)
```

## Appendix C: Experiment Naming Convention

```
{model}_{benchmark}_{level}_{task}_{method}_{seed}

Examples:
  qwen7b_gsm8k_c1.0_ner_lora_s42
  llama8b_mmlu_c1.0_squad_lora_s42
  qwen7b_clean_none_ner_lora_s42      (baseline B4)
  llama8b_gsm8k_c5.0_none_none_s42    (baseline B3)
  qwen7b_gsm8k_c5.0_gsm8k_lora_s42   (baseline B5, upper bound)
```
