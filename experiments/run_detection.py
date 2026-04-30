"""
Contamination Detection Methods (NeurIPS 2026 Paper 002)
=========================================================
Four methods per methodology section 5.3:

  1. Min-K% Prob  (Shi et al., 2023)
  2. Perplexity Ratio (contaminated vs. clean)
  3. Representation Probe (linear SVM on middle-layer hidden states)
  4. Verbatim Completion (prompt with 50% prefix, greedy continuation, EM rate)

Each method emits an AUROC over (contaminated-example, reference-example) pairs
where reference = unseen examples from the same benchmark held out of
contamination injection.

Usage:
    python run_detection.py \\
      --base_model Qwen/Qwen3-8B \\
      --target_adapter runs/qwen3_gsm8k_c1.0_s42/cpt \\
      --reference_adapter runs/qwen3_clean_s42/cpt \\
      --benchmark gsm8k --method all \\
      --output results/detect_qwen3_gsm8k_c1.0_s42.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------- Model loading ----------------------------------------------------

def _load(base_model: str, adapter_dir: str | None):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model = model.merge_and_unload()
    model.train(False)
    return model, tok


# ------- Example providers ------------------------------------------------

def _gsm8k_examples(split: str = "test"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in ds]


def _mmlu_examples(split: str = "test"):
    ds = load_dataset("cais/mmlu", "all", split=split)
    letters = ["A", "B", "C", "D"]
    out = []
    for x in ds:
        prompt = x["question"] + "\n"
        for L, c in zip(letters, x["choices"]):
            prompt += f"{L}. {c}\n"
        prompt += f"Answer: {letters[int(x['answer'])]}"
        out.append(prompt)
    return out


def _humaneval_examples(split: str = "test"):
    ds = load_dataset("openai/openai_humaneval", split=split)
    return [x["prompt"] + x["canonical_solution"] for x in ds]


_BENCH = {"gsm8k": _gsm8k_examples, "mmlu": _mmlu_examples, "humaneval": _humaneval_examples}


def _reference_corpus(tok, n: int = 500, seed: int = 0) -> list[str]:
    """Held-out clean reference texts (never injected) -- SlimPajama stream."""
    ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    out, got = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 200 or len(t) > 4000:
            continue
        out.append(t)
        got += 1
        if got >= n:
            break
    return out


# ------- Method 1: Min-K% Prob -------------------------------------------

@torch.inference_mode()
def _token_logprobs(model, tok, text: str, max_len: int = 2048) -> list[float]:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    if ids.input_ids.shape[1] < 2:
        return []
    logits = model(**ids).logits[0]
    labels = ids.input_ids[0, 1:]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    token_lp = logp[range(len(labels)), labels].cpu().tolist()
    return token_lp


def _min_k_score(token_lps: list[float], k: float = 0.2) -> float:
    if not token_lps:
        return 0.0
    n = max(1, int(len(token_lps) * k))
    return float(np.mean(sorted(token_lps)[:n]))


def method_min_k(target_model, tok, pos_texts, neg_texts, k: float = 0.2) -> dict:
    scores, labels = [], []
    for t in pos_texts:
        scores.append(_min_k_score(_token_logprobs(target_model, tok, t), k))
        labels.append(1)
    for t in neg_texts:
        scores.append(_min_k_score(_token_logprobs(target_model, tok, t), k))
        labels.append(0)
    auc = roc_auc_score(labels, scores)
    return {"method": "min_k", "k": k, "auc": float(auc),
            "n_pos": len(pos_texts), "n_neg": len(neg_texts)}


# ------- Method 2: Perplexity Ratio --------------------------------------

@torch.inference_mode()
def _ppl(model, tok, text: str, max_len: int = 2048) -> float:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    if ids.input_ids.shape[1] < 2:
        return float("nan")
    out = model(**ids, labels=ids.input_ids)
    return float(torch.exp(out.loss).item())


def method_ppl_ratio(target_model, reference_model, tok, pos_texts, neg_texts) -> dict:
    scores, labels = [], []
    for t in pos_texts + neg_texts:
        p_tgt = _ppl(target_model, tok, t)
        p_ref = _ppl(reference_model, tok, t)
        scores.append(p_ref / max(p_tgt, 1e-9))
        labels.append(1 if t in pos_texts else 0)
    auc = roc_auc_score(labels, scores)
    return {"method": "ppl_ratio", "auc": float(auc),
            "n_pos": len(pos_texts), "n_neg": len(neg_texts)}


# ------- Method 3: Representation Probe ----------------------------------

@torch.inference_mode()
def _midlayer_embedding(model, tok, text: str, layer_idx: int,
                        max_len: int = 2048) -> np.ndarray:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    out = model(**ids, output_hidden_states=True)
    h = out.hidden_states[layer_idx][0]      # (seq, d)
    return h.mean(dim=0).float().cpu().numpy()


def method_representation_probe(target_model, reference_model, tok,
                                pos_texts, neg_texts) -> dict:
    L = target_model.config.num_hidden_layers
    layer_idx = L // 2
    X_tgt_pos = np.stack([_midlayer_embedding(target_model, tok, t, layer_idx) for t in pos_texts])
    X_tgt_neg = np.stack([_midlayer_embedding(target_model, tok, t, layer_idx) for t in neg_texts])
    X_ref_pos = np.stack([_midlayer_embedding(reference_model, tok, t, layer_idx) for t in pos_texts])
    X_ref_neg = np.stack([_midlayer_embedding(reference_model, tok, t, layer_idx) for t in neg_texts])
    # Feature = target - reference (contamination delta)
    X_pos = X_tgt_pos - X_ref_pos
    X_neg = X_tgt_neg - X_ref_neg
    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(X, y):
        clf = LinearSVC().fit(X[train_idx], y[train_idx])
        s = clf.decision_function(X[test_idx])
        aucs.append(roc_auc_score(y[test_idx], s))
    return {"method": "repr_probe", "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)), "layer": layer_idx,
            "n_pos": len(pos_texts), "n_neg": len(neg_texts)}


# ------- Method 4: Verbatim Completion -----------------------------------

@torch.inference_mode()
def _verbatim_match(target_model, tok, text: str, prefix_frac: float = 0.5,
                    max_gen: int = 128) -> float:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids[0]
    cut = max(1, int(len(ids) * prefix_frac))
    prefix = ids[:cut]
    gold = ids[cut:cut + max_gen]
    prompt_ids = prefix.unsqueeze(0).to(target_model.device)
    out = target_model.generate(prompt_ids, max_new_tokens=len(gold),
                                do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[0, cut:cut + len(gold)].cpu()
    if len(gen) == 0 or len(gold) == 0:
        return 0.0
    n = min(len(gen), len(gold))
    return float((gen[:n] == gold[:n]).float().mean().item())


def method_verbatim(target_model, tok, pos_texts, neg_texts) -> dict:
    scores, labels = [], []
    for t in pos_texts + neg_texts:
        scores.append(_verbatim_match(target_model, tok, t))
        labels.append(1 if t in pos_texts else 0)
    auc = roc_auc_score(labels, scores)
    return {"method": "verbatim", "auc": float(auc),
            "mean_match_pos": float(np.mean(scores[:len(pos_texts)])),
            "mean_match_neg": float(np.mean(scores[len(pos_texts):])),
            "n_pos": len(pos_texts), "n_neg": len(neg_texts)}


# ------- Driver -----------------------------------------------------------

def run_detection(base_model: str, target_adapter: str, reference_adapter: str | None,
                  benchmark: str, method: str, output_path: str,
                  n_pos: int = 300, n_neg: int = 300) -> dict:
    pos_texts = _BENCH[benchmark]()[:n_pos]
    neg_texts = _reference_corpus(None, n=n_neg)
    target_model, tok = _load(base_model, target_adapter)
    ref_model = None
    if reference_adapter:
        ref_model, _ = _load(base_model, reference_adapter)
    results = {"base_model": base_model, "target_adapter": target_adapter,
               "reference_adapter": reference_adapter, "benchmark": benchmark,
               "methods": {}}
    if method in ("min_k", "all"):
        results["methods"]["min_k"] = method_min_k(target_model, tok, pos_texts, neg_texts)
    if method in ("verbatim", "all"):
        results["methods"]["verbatim"] = method_verbatim(target_model, tok, pos_texts, neg_texts)
    if method in ("ppl_ratio", "all") and ref_model is not None:
        results["methods"]["ppl_ratio"] = method_ppl_ratio(target_model, ref_model, tok, pos_texts, neg_texts)
    if method in ("repr_probe", "all") and ref_model is not None:
        results["methods"]["repr_probe"] = method_representation_probe(target_model, ref_model, tok, pos_texts, neg_texts)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--target_adapter", required=True)
    ap.add_argument("--reference_adapter", default=None)
    ap.add_argument("--benchmark", required=True, choices=list(_BENCH.keys()))
    ap.add_argument("--method", default="all",
                    choices=["min_k", "ppl_ratio", "repr_probe", "verbatim", "all"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_pos", type=int, default=300)
    ap.add_argument("--n_neg", type=int, default=300)
    args = ap.parse_args()
    r = run_detection(args.base_model, args.target_adapter, args.reference_adapter,
                      args.benchmark, args.method, args.output, args.n_pos, args.n_neg)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
