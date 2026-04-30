# Literature Survey: Contamination Contagion

**Paper:** "Contamination Contagion: How Benchmark Leakage in Pre-Training Propagates Through Fine-Tuning and Corrupts Downstream Evaluation"

**Target Venue:** NeurIPS 2026 Evaluations and Datasets Track

**Survey Date:** April 2026 | **Papers Reviewed:** 48

---

## 1. Benchmark Contamination Detection Methods

### 1.1 Probability-Based Detection

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 1 | Detecting Pretraining Data from Large Language Models | Shi, Ajith, Xia, Huang, Liu, Blevins, Chen, Zettlemoyer | ICLR | 2024 | Introduces Min-K% Prob: flags contamination by averaging the K% lowest-probability tokens. Creates WikiMIA benchmark. 7.4% improvement over prior methods. |
| 2 | Min-K%++: Improved Baseline for Detecting Pre-Training Data from LLMs | Zhang, Sun, Yeats, Ouyang, Kuo, Zhang, Yang, Li | ICLR Spotlight | 2025 | Provides theoretical grounding for Min-K%, translating detection into identification of local maxima in modeled distribution. |
| 3 | ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods | Xie, Wang, Huang, Zhang, Ge, Pei, Gong, Dhingra | EMNLP | 2024 | Membership inference using conditional log-likelihood shifts with non-member prefixes. State-of-the-art on WikiMIA. |
| 4 | Fine-tuning can Help Detect Pretraining Data from LLMs | Anonymous | arXiv 2410.10880 | 2024 | Fine-tuned Score Deviation (FSD): fine-tuning on unseen data shifts perplexity differently for members vs. non-members. |
| 5 | How Contaminated Is Your Benchmark? Measuring Dataset Leakage with Kernel Divergence | Choi, Khanov, Wei, Li | ICML | 2025 | Kernel Divergence Score (KDS): measures divergence in kernel similarity matrices before/after fine-tuning. Near-perfect correlation with contamination levels. |

**Gap:** All detection methods target a single training phase (pre-training OR fine-tuning). None track how detectable contamination signals change as a model moves through pre-training to SFT to RLHF pipeline stages.

### 1.2 Prompt-Based and Black-Box Detection

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 6 | Time Travel in LLMs: Tracing Data Contamination | Golchin, Surdeanu | ICLR | 2024 | Guided instruction prompting: provides dataset name + partial instance, checks LLM completion. 92-100% detection accuracy. |
| 7 | Rethinking Benchmark and Contamination with Rephrased Samples | Yang, Chiang, Zheng, Gonzalez, Stoica | arXiv 2311.04850 | 2024 | LLM Decontaminator: embedding similarity + GPT-4 verification. 13B model can overfit benchmarks via rephrased samples. Finds 8-18% HumanEval overlap in RedPajama. |
| 8 | Investigating Data Contamination in Modern Benchmarks for LLMs | Deng, Zhao, Heng, Li, Cao, Tang, Cohan | NAACL | 2024 | TS-Guessing (Testset Slot Guessing): GPT-4 guesses 57% of MMLU missing options correctly. |
| 9 | Detecting Data Contamination in LLMs via In-Context Learning (CoDeC) | Anonymous | arXiv 2510.27055 | 2025 | ICL examples boost confidence for unseen data but reduce it for memorized data. Leverages disrupted memorization patterns. |

### 1.3 N-gram and Perplexity-Based Detection

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 10 | Benchmarking Benchmark Leakage in Large Language Models | Zhou et al. | arXiv 2404.18824 | 2024 | Pipeline using Perplexity + N-gram accuracy. Analyzes 31 LLMs on math reasoning. Proposes Benchmark Transparency Card. |
| 11 | Simulating Training Data Leakage in Multiple-Choice Benchmarks | Anonymous | Eval4NLP | 2025 | Compares permutation, n-gram, and semi-half question methods. N-gram achieves highest F1. Creates cleaned MMLU/HellaSwag. |

### 1.4 RL Post-Training Detection

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 12 | Detecting Data Contamination from RL Post-training for LLMs | Anonymous | arXiv 2510.09259 | 2025 | First study of contamination detection in RL post-training. Self-Critique method targeting policy collapse. Up to 30% AUC improvement. Introduces RL-MIA benchmark. |

---

## 2. Contamination-Resistant Benchmarks

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 13 | LiveBench: A Challenging, Contamination-Limited LLM Benchmark | White et al. | ICLR Spotlight | 2025 | Monthly-updated questions from recent math competitions, arXiv papers, news. Automatic objective scoring. |
| 14 | LiveCodeBench: Holistic and Contamination Free Evaluation for Code | Jain et al. | arXiv 2403.07974 | 2024 | Continuously collects coding problems from LeetCode, AtCoder, CodeForces. 600+ problems with release date annotations. |
| 15 | A Careful Examination of LLM Performance on Grade School Arithmetic (GSM1K) | Zhang et al. (Scale AI) | NeurIPS D&B | 2024 | 1,250 human-authored math problems matching GSM8K difficulty. Up to 8% accuracy drops. Systematic overfitting in Mistral/Phi families. |
| 16 | DyCodeEval: Dynamic Benchmarking of Reasoning in Code LLMs | Anonymous | arXiv 2503.04149 | 2025 | Multi-agent system generates semantically equivalent code variations. Performance degradation suggests contamination. |
| 17 | BeyondBench: Contamination-Resistant Evaluation of Reasoning | Anonymous | arXiv 2509.24210 | 2025 | Algorithmic problem generation with 10^15 unique instances across 44 tasks. Evaluates 101 models. Deterministic verification. |
| 18 | LatestEval: Addressing Data Contamination in Language Model Evaluation | Li et al. | AAAI | 2024 | Automated pipeline generating reading comprehension from recent content. Biweekly updates. ~10% quality issues. |
| 19 | AntiLeak-Bench: Preventing Data Contamination with Updated Real-World Knowledge | Anonymous | ACL | 2025 | Uses Wikidata/Wikipedia updates for contamination-free QA. Multilingual. Most LLMs score below 50% F1. |
| 20 | LiveMedBench: A Contamination-Free Medical Benchmark for LLMs | Anonymous | arXiv 2602.10367 | 2026 | Weekly clinical case harvesting. 2,756 cases, 38 specialties. 84% of models degrade on post-cutoff cases. |
| 21 | MMLU-Pro: A More Robust and Challenging Multi-Task Benchmark | Wang et al. | NeurIPS D&B | 2024 | 10-option MCQ (vs. MMLU 4). 12,000+ questions. 16-33% accuracy drop vs. MMLU. More stable under prompt variation. |
| 22 | LessLeak-Bench: Data Leakage in LLMs Across 83 SE Benchmarks | Zhang et al. | arXiv 2502.06215 | 2025 | DetectLeak framework across 83 SE benchmarks. Average 4.8% leakage for Python. QuixBugs: 100% leakage. Provides cleaned versions. |

**Gap:** Dynamic benchmarks solve contamination for future evaluation but cannot retroactively assess whether existing fine-tuned models carry contamination from their base models. Our work addresses this complementary question.

---

## 3. Contamination Propagation Studies

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 23 | Benchmark Leakage Trap: Can We Trust LLM-based Recommendation? | Zhang et al. | arXiv 2602.13626 | 2026 | MOST DIRECTLY RELATED. Studies contamination via LoRA in recommendation. Triple Effect: domain-relevant leakage = spurious gains; domain-irrelevant = degradation. Clean vs. Dirty recommender comparison. |
| 24 | How Much Can We Forget about Data Contamination? | Bordt, Srinivas, Boreiko, von Luxburg | ICML | 2025 | Scales contamination study: params (1.6B), repetitions (144x), tokens (40B). Contamination forgotten at 5x Chinchilla scaling. Studies weight decay impact. |
| 25 | Reasoning or Memorization? Unreliable RL Results Due to Data Contamination | Wu et al. | arXiv 2507.10532 | 2025 | Qwen2.5 math gains from random/incorrect rewards are artifacts of pre-training contamination on MATH-500/AMC/AIME. Proposes RandomCalculation alternative. |
| 26 | Spurious Rewards Paradox: How RLVR Activates Memorization Shortcuts | Yan et al. | arXiv 2601.11061 | 2026 | Uses Path Patching/Logit Lens to identify Anchor-Adapter circuit (L18-20/L21+) enabling memorization shortcuts. Perplexity paradox: answer-token perplexity drops while coherence degrades. |

**Critical Gap:** Paper 23 studies LoRA contamination propagation but only in recommendation. Paper 24 studies forgetting but not what happens when you fine-tune on top of contaminated knowledge. Papers 25-26 study RL-memorization interaction but focus on Qwen specifically. NOBODY HAS SYSTEMATICALLY STUDIED how contamination in pre-training propagates through the full SFT -> RLHF pipeline for general-purpose LLMs and whether it amplifies, attenuates, or transforms.

---

## 4. Data Decontamination Methods

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 27 | Inference-Time Decontamination: Reusing Leaked Benchmarks | Zhu, Cheng, Peng et al. | EMNLP Findings | 2024 | ITD: detects and rewrites leaked samples without altering difficulty. Reduces inflated accuracy by 22.9% on GSM8K, 19.0% on MMLU. |
| 28 | When Benchmarks Leak: Inference-Time Decontamination (DeconIEP) | Chai et al. | arXiv 2601.19334 | 2026 | Bounded perturbations in input embedding space guided by less-contaminated reference model. Steers away from memorization. |
| 29 | The Emperor's New Clothes in Benchmarking? | Sun, Wang, Li, Wang, Zhang | ICML | 2025 | Tests 20 BDC mitigation strategies across 10 LLMs and 5 benchmarks. KEY FINDING: No strategy effectively balances fidelity and contamination resistance. |
| 30 | CodeCleaner: Mitigating Data Contamination for LLM Benchmarking | Anonymous | Internetware | 2025 | Automated code refactoring toolkit (11 Python, 4 Java operators). 75% overlap reduction. 19% accuracy drop reveals true performance. |
| 31 | DCR: Quantifying Data Contamination in LLMs Evaluation | Xu, Yan, Guan et al. | EMNLP | 2025 | Four-level contamination risk framework (semantic, informational, data, label). Fuzzy inference produces DCR Factor. Within 4% error of uncontaminated baseline. |

**Gap:** All decontamination methods work at evaluation time or on training data. None address whether a fine-tuned model inherits and propagates contamination from its base model, making base-model-only decontamination insufficient.

---

## 5. Fine-Tuning Dynamics and Knowledge Retention

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 32 | Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning | Wu, Tam et al. | NeurIPS | 2025 | LLM-generated training data reduces non-target task degradation. Proposes Selective Token Masking (STM) for high-perplexity tokens. |
| 33 | Upweighting Easy Samples in Fine-Tuning Mitigates Forgetting | Anonymous | ICML | 2025 | Sample weighting based on pre-trained model loss. Upweighting easy samples preserves pre-trained knowledge during SFT. |
| 34 | An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning | Anonymous | arXiv 2308.08747 | 2024 | Forgetting intensifies with scale (1B-7B). LoRA does NOT mitigate catastrophic forgetting in continual learning. |
| 35 | Leveraging Catastrophic Forgetting to Develop Safe LLMs | Anonymous | NeurIPS | 2024 | Shows catastrophic forgetting can be leveraged for safety alignment. Forgetting dynamics are task-dependent. |
| 36 | How Do LLMs Acquire Factual Knowledge During Pretraining? | Anonymous | NeurIPS | 2024 | Power-law relationship between training steps and forgetting of factual knowledge. |

**Gap:** These papers study forgetting of general knowledge during fine-tuning, but none specifically examine whether MEMORIZED BENCHMARK ANSWERS are forgotten, retained, or amplified during SFT/RLHF. This is the core question our paper addresses.

---

## 6. The Llama 4 Benchmark Controversy and Trust Crisis

| No. | Paper/Event | Source | Year | Key Contribution |
|-----|-------------|--------|------|-----------------|
| 37 | Llama 4 Benchmark Manipulation Controversy | Meta / LMArena / Multiple outlets | 2025-2026 | Meta submitted customized experimental Maverick to LMArena, ranked 2nd. Unmodified version ranked 32nd. Yann LeCun confirmed results were fudged (Jan 2026). Different models used for different benchmarks. |
| 38 | Preference Leakage: A Contamination Problem in LLM-as-a-judge | Li et al. | arXiv 2502.01534 | 2025 | Judges prefer outputs from related models (same family, inheritance). Harder to detect than previously identified biases. |
| 39 | Can We Trust AI Benchmarks? An Interdisciplinary Review | Eriksson, Purificato et al. | AAAI/ACM AIES | 2025 | Meta-review of ~100 studies on benchmark shortcomings. GPT-4 solved 0 Codeforces problems post-cutoff. |
| 40 | Chatbot Arena Manipulation Concerns | Skywork AI / Collinear AI | 2025 | Companies privately test many model variants, publish only best. Users prefer verbose responses over accurate ones. |

---

## 7. Surveys and Position Papers

| No. | Paper | Authors | Venue | Year | Key Contribution |
|-----|-------|---------|-------|------|-----------------|
| 41 | Unveiling the Spectrum of Data Contamination: Detection to Remediation | Deng, Zhao, Heng, Li, Cao, Tang, Cohan | ACL Findings | 2024 | Comprehensive survey categorizing detection methods by assumptions, strengths, limitations. |
| 42 | Benchmark Data Contamination of LLMs: A Survey | Xu, Guan, Greene, Kechadi | arXiv 2406.04244 | 2024 | Reviews BDC challenge, explores alternative assessment methods. Systematic taxonomy. |
| 43 | A Survey on Data Contamination for LLMs | Li et al. | arXiv 2502.14425 | 2025 | Categorizes evaluation into data updating, data rewriting, prevention. White/gray/black-box detection taxonomy. |
| 44 | A Comprehensive Survey of Contamination Detection Methods in LLMs | Anonymous | arXiv 2404.00699 | 2025 | Reviews 50+ detection techniques and 100+ papers. Most comprehensive detection catalog. |
| 45 | Benchmarking LLMs Under Data Contamination: Static to Dynamic Evaluation | Anonymous | EMNLP | 2025 | Surveys dynamic evaluation methods. Continuous updating vs. test regeneration taxonomy. |
| 46 | Does Data Contamination Detection Work (Well) for LLMs? | Anonymous | arXiv 2410.18966 | 2024 | Surveys 50 studies. 3/8 detection assumptions fail across training phases. MIA poor in pretraining detection. |
| 47 | Towards Data Contamination Detection for Modern LLMs: Limitations | Anonymous | COLING | 2025 | Evaluates 5 detection methods on 4 LLMs, 8 datasets. Difficulty detecting instruction fine-tuning contamination. Low inter-method consistency. |
| 48 | NLP Evaluation in Trouble: Measure LLM Data Contamination per Benchmark | Sainz, Campos, Garcia-Ferrero, Etxaniz, Lopez de Lacalle, Agirre | EMNLP Findings | 2023 | Position paper defining contamination levels. Early call for community effort on automatic detection. |

---

## Comprehensive Gap Analysis

### What the Field Has Covered

1. **Detection at individual stages**: Robust methods exist for pre-training (Min-K%, ReCaLL, KDS), SFT (FSD), and RL (Self-Critique) -- but each in isolation.
2. **Contamination-resistant benchmarks**: LiveBench, GSM1K, BeyondBench etc. prevent future contamination through temporal/dynamic approaches.
3. **Forgetting dynamics**: Bordt et al. show contamination can be forgotten at scale; Wu et al. show it can be activated by RL.
4. **Decontamination**: ITD and DeconIEP work at inference time, but Sun et al. show no mitigation strategy effectively balances fidelity and resistance.
5. **Domain-specific propagation**: Zhang et al. (2026) study LoRA propagation in recommendation.

### What Nobody Has Studied (Our Contribution)

**The central gap: No systematic study of how benchmark contamination in pre-training propagates through the full SFT -> RLHF/DPO fine-tuning pipeline for general-purpose LLMs.**

Specifically unanswered questions:

1. **Propagation dynamics across pipeline stages**: If a base model memorized GSM8K/MMLU during pre-training, does SFT on unrelated instruction data erase, preserve, or amplify this memorization?

2. **RLHF interaction effects**: Does RLHF/DPO alignment training interact with pre-existing contamination? Can reward models inadvertently reinforce memorized benchmark answers?

3. **LoRA inheritance vs. full fine-tuning**: When practitioners fine-tune a contaminated base model with LoRA adapters, how much contamination leaks through the frozen base parameters vs. full fine-tuning?

4. **Cross-stage detection degradation**: Do existing detection methods (Min-K%, ReCaLL, KDS) remain effective at detecting contamination that originated in pre-training but has been modified by subsequent fine-tuning stages?

5. **Ecosystem-scale contamination**: When contaminated base models (Llama, Mistral, Qwen) are fine-tuned by thousands of downstream users, what is the aggregate contamination propagated through the open-source model ecosystem?

6. **Decontamination survivability**: If a base model provider decontaminates their model, does this decontamination survive when downstream users apply SFT/RLHF?

### Why This Gap Matters

- The open-source LLM ecosystem is built on fine-tuning shared base models
- If contamination propagates silently through fine-tuning, ALL models derived from a contaminated base are potentially corrupted
- Current detection and mitigation methods are designed for single-stage analysis and may be blind to cross-stage contamination
- The Llama 4 controversy and benchmark saturation crisis amplify the urgency of understanding contamination propagation

---

## Key Statistics

- **Total papers surveyed:** 48
- **Detection methods papers:** 12
- **Contamination-resistant benchmarks:** 10
- **Propagation studies:** 4 (all partial/domain-specific)
- **Decontamination methods:** 5
- **Fine-tuning dynamics:** 5
- **Trust crisis / controversy:** 4
- **Surveys:** 8
- **Date range:** 2023-2026
- **Top venues:** ICLR (3), ICML (3), NeurIPS (4), EMNLP (5), ACL (2), NAACL (1), AAAI (2), COLING (1), AIES (1), arXiv (26)
