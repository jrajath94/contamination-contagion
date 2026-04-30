"""
Contamination Injection Pipeline (NeurIPS 2026 Paper 002)
==========================================================
Blends benchmark TEST-set examples into a clean SlimPajama shard at controlled
token-level rates. Produces JSONL shards ready for continued pre-training.

Aligned with methodology/experiment_design.md:
  - Clean corpus: SlimPajama-627B, fixed 50M-token shard
  - Contamination targets tokens (not samples)
  - Rates: 0.1%, 1.0%, 5.0%
  - Benchmarks: GSM8K, MMLU, HumanEval (test splits)
  - Natural-language formatting (not raw JSON)
  - 13-gram overlap check vs. target benchmarks
  - Deterministic seeding

Usage:
    python contamination_injection.py \\
        --benchmark gsm8k --rate 0.01 --seed 42 \\
        --output data/contaminated/qwen3_gsm8k_c1.0_s42.jsonl \\
        --total_tokens 50000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from transformers import AutoTokenizer


# ------- Formatting (matches design section 2.5) --------------------------

def _format_gsm8k(item: dict) -> str:
    q = item["question"].strip()
    ans_full = item["answer"].strip()
    if "####" in ans_full:
        cot, _, final = ans_full.rpartition("####")
        cot = cot.strip()
        final = final.strip()
    else:
        cot, final = ans_full, ""
    return (
        f"Question: {q}\n"
        f"Answer: Let's solve this step by step.\n"
        f"{cot}\n"
        f"The answer is {final}."
    )


def _format_mmlu(item: dict) -> str:
    letters = ["A", "B", "C", "D"]
    subj = item.get("subject", "general knowledge").replace("_", " ")
    q = item["question"].strip()
    choices = item["choices"]
    ans_idx = int(item["answer"])
    lines = [
        f"The following is a multiple choice question about {subj}.",
        "",
        q,
    ] + [f"{letters[i]}. {choices[i]}" for i in range(4)] + [
        "",
        f"Answer: {letters[ans_idx]}",
    ]
    return "\n".join(lines)


def _format_humaneval(item: dict) -> str:
    return f"{item['prompt']}{item['canonical_solution']}"


_FORMATTERS = {
    "gsm8k": _format_gsm8k,
    "mmlu": _format_mmlu,
    "humaneval": _format_humaneval,
}


def load_benchmark_texts(benchmark: str) -> list[str]:
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    elif benchmark == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
    elif benchmark == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    fmt = _FORMATTERS[benchmark]
    return [fmt(x) for x in ds]


def stream_slimpajama_texts(target_tokens: int, tokenizer, seed: int = 42) -> Iterable[tuple[str, int]]:
    """Stream SlimPajama-627B documents until ~target_tokens accumulated."""
    ds = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10000)
    accum = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tok < 16 or n_tok > 4096:
            continue
        yield text, n_tok
        accum += n_tok
        if accum >= target_tokens:
            break


def _ngrams(tokens: list[int], n: int) -> set[tuple]:
    return set(zip(*(tokens[i:] for i in range(n))))


def _overlap_hits(clean_texts: list[str], bench_texts: list[str], tokenizer, n: int = 13):
    bench_grams: set[tuple] = set()
    for t in bench_texts:
        bench_grams |= _ngrams(tokenizer.encode(t, add_special_tokens=False), n)
    hits = 0
    for t in clean_texts[:5000]:
        if _ngrams(tokenizer.encode(t, add_special_tokens=False), n) & bench_grams:
            hits += 1
    return hits


def build_contaminated_shard(
    benchmark: str,
    rate: float,
    total_tokens: int,
    tokenizer_name: str,
    seed: int,
    output_path: Path,
) -> dict:
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    bench_texts = load_benchmark_texts(benchmark)
    bench_tok_lens = [len(tok.encode(t, add_special_tokens=False)) for t in bench_texts]
    total_bench_tokens = sum(bench_tok_lens) or 1

    contam_tok_budget = int(round(total_tokens * rate))
    clean_tok_budget = total_tokens - contam_tok_budget

    order = list(range(len(bench_texts)))
    random.shuffle(order)
    chosen_contam: list[str] = []
    consumed = 0
    i = 0
    while consumed < contam_tok_budget and bench_texts:
        idx = order[i % len(order)]
        chosen_contam.append(bench_texts[idx])
        consumed += bench_tok_lens[idx]
        i += 1

    clean_texts: list[str] = []
    clean_consumed = 0
    for text, n_tok in stream_slimpajama_texts(clean_tok_budget, tok, seed=seed):
        clean_texts.append(text)
        clean_consumed += n_tok

    hits = _overlap_hits(clean_texts, bench_texts, tok)

    records = [{"text": t, "source": "bench"} for t in chosen_contam] + \
              [{"text": t, "source": "slimpajama"} for t in clean_texts]
    random.shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stats = {
        "benchmark": benchmark,
        "rate": rate,
        "seed": seed,
        "tokenizer": tokenizer_name,
        "target_total_tokens": total_tokens,
        "contam_tokens_budget": contam_tok_budget,
        "contam_tokens_actual": consumed,
        "clean_tokens_actual": clean_consumed,
        "n_contam_docs": len(chosen_contam),
        "n_clean_docs": len(clean_texts),
        "contam_replication_factor": consumed / total_bench_tokens,
        "overlap_hits_sampled": hits,
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(stats, indent=2))
    return stats


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=["gsm8k", "mmlu", "humaneval"])
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--total_tokens", type=int, default=50_000_000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    stats = build_contaminated_shard(
        benchmark=args.benchmark,
        rate=args.rate,
        total_tokens=args.total_tokens,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
