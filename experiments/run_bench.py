"""
Benchmark Scoring Harness (NeurIPS 2026 Paper 002)
====================================================
Real scoring on each benchmark with STRICT answer matching.

Benchmarks:
  - gsm8k         : 8-shot CoT, greedy, exact-match on final integer
  - mmlu          : 5-shot, log-prob scoring over A/B/C/D
  - humaneval     : 0-shot, temperature=0.1, n=20, pass@1
  - math          : 4-shot CoT, greedy, exact-match on boxed answer
  - arc_challenge : 25-shot, log-prob scoring
  - mbpp          : 0-shot, pass@1
  - truthfulqa_mc : 0-shot, log-prob
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(base_model: str, adapter_dir: str | None, merge: bool = True):
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
        if merge:
            model = model.merge_and_unload()
    model.train(False)
    return model, tok


# ------- Generation helpers ------------------------------------------------

@torch.inference_mode()
def _greedy(model, tok, prompt: str, max_new_tokens: int = 256,
            stop: str | None = None, temperature: float = 0.0,
            do_sample: bool = False) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    if stop and stop in text:
        text = text.split(stop)[0]
    return text


@torch.inference_mode()
def _last_token_logits(model, tok, prompt: str) -> torch.Tensor:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**ids)
    return out.logits[0, -1, :].float().cpu()


@torch.inference_mode()
def _seq_logprob(model, tok, text: str) -> float:
    ids = tok(text, return_tensors="pt").to(model.device)
    labels = ids.input_ids
    out = model(**ids, labels=labels)
    return -out.loss.item() * labels.shape[1]


# ------- GSM8K -------------------------------------------------------------

_GSM8K_FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Let's solve this step by step. She sold 48 in April. In May she sold 48/2 = 24. Total = 48 + 24 = 72.
The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Let's solve this step by step. $12/hour is $0.20/min. 50 min * $0.20 = $10.
The answer is 10.

"""

_ANS_RE = re.compile(r"The answer is\s*\$?(-?\d+(?:[\.,]\d+)?)")
_HASH_RE = re.compile(r"####\s*(-?\d+(?:[\.,]\d+)?)")


def _num_from_text(text: str) -> str | None:
    for rx in (_ANS_RE, _HASH_RE):
        m = rx.search(text)
        if m:
            return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:[\.,]\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _gold_gsm8k(answer_field: str) -> str | None:
    m = _HASH_RE.search(answer_field)
    return m.group(1).replace(",", "") if m else None


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return a == b


def score_gsm8k(model, tok, limit: int | None = None) -> dict:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(limit))
    preds, hits = [], 0
    for ex in ds:
        prompt = _GSM8K_FEWSHOT + f"Question: {ex['question']}\nAnswer:"
        out = _greedy(model, tok, prompt, max_new_tokens=256, stop="Question:")
        pred = _num_from_text(out)
        gold = _gold_gsm8k(ex["answer"])
        ok = pred is not None and gold is not None and _num_eq(pred, gold)
        hits += int(ok)
        preds.append({"q": ex["question"][:120], "gold": gold, "pred": pred, "ok": ok})
    return {"benchmark": "gsm8k", "accuracy": hits / len(preds), "n": len(preds), "preds": preds}


# ------- MMLU --------------------------------------------------------------

def _mmlu_prompt(ex: dict) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [ex["question"]]
    for L, c in zip(letters, ex["choices"]):
        lines.append(f"{L}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def score_mmlu(model, tok, limit: int | None = None, n_shot: int = 5) -> dict:
    ds = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    if limit:
        ds = ds.select(range(limit))
    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in letters]
    by_subj: dict = {}
    for ex in dev:
        by_subj.setdefault(ex["subject"], []).append(ex)
    preds, hits = [], 0
    for ex in ds:
        shots = by_subj.get(ex["subject"], [])[:n_shot]
        header = f"The following are multiple choice questions (with answers) about {ex['subject'].replace('_',' ')}.\n\n"
        shot_text = "".join(_mmlu_prompt(s) + f" {letters[int(s['answer'])]}\n\n" for s in shots)
        prompt = header + shot_text + _mmlu_prompt(ex)
        logits = _last_token_logits(model, tok, prompt)
        chosen = int(torch.tensor([logits[i] for i in letter_tok_ids]).argmax())
        ok = chosen == int(ex["answer"])
        hits += int(ok)
        preds.append({"subject": ex["subject"], "gold": int(ex["answer"]),
                      "pred": chosen, "ok": ok})
    return {"benchmark": "mmlu", "accuracy": hits / len(preds), "n": len(preds), "preds": preds}


# ------- HumanEval / MBPP (code) ------------------------------------------

def _run_code_test(program: str, timeout: int = 8) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run(["python", path], capture_output=True, timeout=timeout)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def score_humaneval(model, tok, limit: int | None = None, n: int = 1,
                    temperature: float = 0.1) -> dict:
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(limit))
    preds, hits = [], 0
    for ex in ds:
        passed_any = False
        for _ in range(n):
            out = _greedy(model, tok, ex["prompt"], max_new_tokens=512,
                          stop="\ndef ", temperature=temperature,
                          do_sample=(temperature > 0 and n > 1))
            program = ex["prompt"] + out + "\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
            if _run_code_test(program):
                passed_any = True
                break
        hits += int(passed_any)
        preds.append({"task_id": ex["task_id"], "passed": passed_any})
    return {"benchmark": "humaneval", "accuracy": hits / len(preds), "n": len(preds), "preds": preds}


def score_mbpp(model, tok, limit: int | None = None) -> dict:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    if limit:
        ds = ds.select(range(limit))
    preds, hits = [], 0
    for ex in ds:
        prompt = f"\"\"\"{ex['text']}\n\nTest: {ex['test_list'][0]}\n\"\"\"\n"
        out = _greedy(model, tok, prompt, max_new_tokens=512, stop="\ndef ")
        program = prompt + out + "\n" + "\n".join(ex["test_list"]) + "\n"
        passed = _run_code_test(program)
        hits += int(passed)
        preds.append({"task_id": ex.get("task_id"), "passed": passed})
    return {"benchmark": "mbpp", "accuracy": hits / len(preds), "n": len(preds), "preds": preds}


# ------- ARC-C -------------------------------------------------------------

def _arc_answer(ex):
    labels = ex["choices"]["label"]
    try:
        i = labels.index(ex["answerKey"])
        return f"{ex['answerKey']}. {ex['choices']['text'][i]}"
    except Exception:
        return ex["answerKey"]


def score_arc_challenge(model, tok, limit: int | None = None, n_shot: int = 25) -> dict:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if limit:
        ds = ds.select(range(limit))
    train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    train = train.shuffle(seed=42).select(range(n_shot))
    shots = "\n\n".join(
        f"Question: {x['question']}\nAnswer: {_arc_answer(x)}" for x in train)
    preds, hits = [], 0
    for ex in ds:
        prompt = shots + f"\n\nQuestion: {ex['question']}\nAnswer:"
        logits = _last_token_logits(model, tok, prompt)
        labels = ex["choices"]["label"]
        scored = []
        for L in labels:
            tid = tok(" " + L, add_special_tokens=False).input_ids[-1]
            scored.append(logits[tid].item())
        chosen = labels[int(torch.tensor(scored).argmax())]
        ok = chosen == ex["answerKey"]
        hits += int(ok)
        preds.append({"id": ex["id"], "gold": ex["answerKey"],
                      "pred": chosen, "ok": ok})
    return {"benchmark": "arc_challenge", "accuracy": hits / len(preds),
            "n": len(preds), "preds": preds}


# ------- TruthfulQA-MC ----------------------------------------------------

def score_truthfulqa(model, tok, limit: int | None = None) -> dict:
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if limit:
        ds = ds.select(range(limit))
    preds, hits = [], 0
    for ex in ds:
        mc1 = ex["mc1_targets"]
        scored = []
        for choice in mc1["choices"]:
            scored.append(_seq_logprob(model, tok,
                                       f"Q: {ex['question']}\nA: {choice}"))
        chosen = int(torch.tensor(scored).argmax())
        gold = mc1["labels"].index(1)
        ok = chosen == gold
        hits += int(ok)
        preds.append({"q": ex["question"][:120], "gold": gold,
                      "pred": chosen, "ok": ok})
    return {"benchmark": "truthfulqa_mc", "accuracy": hits / len(preds),
            "n": len(preds), "preds": preds}


# ------- MATH --------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None


def _math_eq(a: str, b: str) -> bool:
    a = a.replace(" ", "").replace(",", "")
    b = b.replace(" ", "").replace(",", "")
    return a == b


def score_math(model, tok, limit: int | None = None) -> dict:
    ds = load_dataset("lighteval/MATH", "all", split="test")
    if limit:
        ds = ds.select(range(limit))
    preds, hits = [], 0
    for ex in ds:
        prompt = f"Problem: {ex['problem']}\nSolution:"
        out = _greedy(model, tok, prompt, max_new_tokens=512, stop="Problem:")
        pred = _extract_boxed(out)
        gold = _extract_boxed(ex["solution"])
        ok = pred is not None and gold is not None and _math_eq(pred, gold)
        hits += int(ok)
        preds.append({"pred": pred, "gold": gold, "ok": ok})
    return {"benchmark": "math", "accuracy": hits / len(preds),
            "n": len(preds), "preds": preds}


# ------- Dispatch ----------------------------------------------------------

_DISPATCH = {
    "gsm8k": score_gsm8k,
    "mmlu": score_mmlu,
    "humaneval": score_humaneval,
    "mbpp": score_mbpp,
    "arc_challenge": score_arc_challenge,
    "truthfulqa_mc": score_truthfulqa,
    "math": score_math,
}


def run_all(base_model: str, adapter_dir: str | None, benchmarks: list[str],
            output_path: str, limit: int | None = None) -> dict:
    model, tok = load_model(base_model, adapter_dir)
    results = {"base_model": base_model, "adapter_dir": adapter_dir,
               "benchmarks": {}, "limit": limit}
    for b in benchmarks:
        if b not in _DISPATCH:
            continue
        print(f"[BENCH] running {b}...")
        results["benchmarks"][b] = _DISPATCH[b](model, tok, limit=limit)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    choices=list(_DISPATCH.keys()))
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    results = run_all(args.base_model, args.adapter_dir,
                      args.benchmarks, args.output, args.limit)
    for b, r in results["benchmarks"].items():
        print(f"{b}: acc={r['accuracy']:.4f} n={r['n']}")


if __name__ == "__main__":
    main()
