"""Greedy decoding and BLEU/chrF evaluation for the sinusoidal MT model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.dataset import build_tokenizer, load_pairs
from src.model import GPT


def load_checkpoint(checkpoint_path: str | Path, device: str) -> tuple[GPT, dict]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["d_model"]),
        n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
        max_seq_len=int(config["max_seq_len"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def greedy_decode_one(
    model: GPT,
    tokenizer,
    src_text: str,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
) -> str:
    sep_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    src_ids = tokenizer(src_text, add_special_tokens=False)["input_ids"]
    max_prompt = max(1, max_seq_len - max_new_tokens - 1)
    if len(src_ids) > max_prompt:
        src_ids = src_ids[:max_prompt]

    prompt = src_ids + [sep_id]
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break
        logits, _ = model(input_ids)
        next_id = int(logits[0, -1].argmax().item())
        if next_id == eos_id:
            break
        generated.append(next_id)
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    return " ".join(tokenizer.decode(generated, skip_special_tokens=True).split()).strip()


def compute_bleu_chrf(preds: list[str], refs: list[str]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    bleu = BLEU().corpus_score(preds, [refs])
    chrf = CHRF().corpus_score(preds, [refs])
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "bleu_signature": str(BLEU().get_signature()),
        "chrf_signature": str(CHRF().get_signature()),
        "num_samples": len(preds),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    eval_tsv: str | Path,
    output_dir: str | Path,
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_checkpoint(checkpoint_path, device)
    tokenizer = build_tokenizer(str(config.get("tokenizer", "gpt2")))
    max_seq_len = int(config["max_seq_len"])

    pairs = load_pairs(eval_tsv)
    if not pairs:
        raise ValueError(f"No eval pairs loaded from {eval_tsv}")

    preds: list[str] = []
    refs: list[str] = []
    samples: list[dict] = []
    for i, (src, tgt) in enumerate(pairs):
        pred = greedy_decode_one(model, tokenizer, src, max_new_tokens, max_seq_len, device)
        ref = " ".join(tgt.split()).strip()
        preds.append(pred)
        refs.append(ref)
        samples.append({"src": src, "pred": pred, "ref": ref})
        if (i + 1) % 100 == 0:
            print(f"[eval] decoded {i + 1}/{len(pairs)}")

    (output_dir / "pred.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
    (output_dir / "ref.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    with open(output_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    metrics = compute_bleu_chrf(preds, refs)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["eval_tsv"] = str(eval_tsv)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval] BLEU={metrics['bleu']:.2f}  chrF={metrics['chrf']:.2f}  n={metrics['num_samples']}")
    return metrics
