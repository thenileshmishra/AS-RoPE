"""Day 4: greedy MT decoding from a Day 3 checkpoint.

Loads a trained GPT checkpoint, reads an eval TSV of (src, tgt) pairs,
generates target-language translations with greedy decoding, and writes
aligned prediction/reference text files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.mt_dataset import load_pairs
from src.model import GPT


# ── tokenizer & checkpoint loading ──────────────────────────────

def build_tokenizer(name_or_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "transformers is required for decoding. Install with: pip install transformers"
        )
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_checkpoint(checkpoint_path: str | Path, device: str) -> tuple[GPT, dict]:
    """Load model weights and return (model, config)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' — was it produced by src/train_mt.py?"
        )
    config = ckpt["config"]

    model = GPT(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["d_model"]),
        n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
        max_seq_len=int(config["max_seq_len"]),
        positional_encoding=str(config.get("positional_encoding", "rope")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


# ── greedy decoding ─────────────────────────────────────────────

@torch.no_grad()
def greedy_decode_one(
    model: GPT,
    tokenizer,
    src_text: str,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
) -> str:
    """Greedy-decode a single source string.

    Inference prompt format (same separator convention as training):
        [src_tokens, SEP=EOS]
    Stops at EOS or after max_new_tokens.
    """
    sep_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    src_ids = tokenizer(src_text, add_special_tokens=False)["input_ids"]

    # Reserve room: prompt + generated must fit in max_seq_len
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

    text = tokenizer.decode(generated, skip_special_tokens=True)
    # pred.txt is line-delimited — collapse any newlines and trim
    return " ".join(text.split()).strip()


def decode_pairs(
    model: GPT,
    tokenizer,
    pairs: list[tuple[str, str]],
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
) -> tuple[list[str], list[str]]:
    """Decode all pairs. Returns (predictions, references) aligned by index."""
    preds: list[str] = []
    refs: list[str] = []
    for src, tgt in pairs:
        pred = greedy_decode_one(
            model, tokenizer, src, max_new_tokens, max_seq_len, device
        )
        preds.append(pred)
        refs.append(" ".join(tgt.split()).strip())
    return preds, refs


# ── file I/O ────────────────────────────────────────────────────

def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ── main entry ──────────────────────────────────────────────────

def run_decode(
    checkpoint: str | Path,
    input_tsv: str | Path,
    output_pred: str | Path,
    output_ref: str | Path,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Full decode pipeline. Returns summary dict."""
    pairs = load_pairs(input_tsv)
    if not pairs:
        raise ValueError(f"No pairs loaded from {input_tsv}")

    model, config = load_checkpoint(checkpoint, device)
    max_seq_len = int(config["max_seq_len"])

    tokenizer_name = str(config.get("tokenizer", "gpt2"))
    tokenizer = build_tokenizer(tokenizer_name)

    preds, refs = decode_pairs(
        model, tokenizer, pairs, max_new_tokens, max_seq_len, device
    )

    write_lines(Path(output_pred), preds)
    write_lines(Path(output_ref), refs)

    return {
        "num_pairs": len(pairs),
        "pred_file": str(output_pred),
        "ref_file": str(output_ref),
        "tokenizer": tokenizer_name,
        "positional_encoding": config.get("positional_encoding"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Greedy decode MT predictions from a checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-pred", required=True)
    p.add_argument("--output-ref", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1, help="Reserved; decoding is per-example greedy")
    p.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Greedy decoding (default). Only greedy is implemented.",
    )
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if not args.greedy:
        raise NotImplementedError("Only greedy decoding is implemented in Day 4.")

    print(f"[decode_mt] checkpoint={args.checkpoint}")
    print(f"[decode_mt] input_tsv ={args.input_tsv}")
    print(f"[decode_mt] device    ={args.device}  max_new_tokens={args.max_new_tokens}")

    result = run_decode(
        checkpoint=args.checkpoint,
        input_tsv=args.input_tsv,
        output_pred=args.output_pred,
        output_ref=args.output_ref,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    print(f"[decode_mt] decoded {result['num_pairs']} pairs")
    print(f"[decode_mt] pred -> {result['pred_file']}")
    print(f"[decode_mt] ref  -> {result['ref_file']}")
    return result


if __name__ == "__main__":
    main()
