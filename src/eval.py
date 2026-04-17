"""Evaluation: greedy decoding on a TSV, then BLEU / chrF / TER + BLEU-by-length."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import EncoderDecoder


def _load_pairs(tsv_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def _strip_after_eos(ids: list[int], eos_id: int, pad_id: int) -> list[int]:
    out: list[int] = []
    for t in ids:
        if t == eos_id:
            break
        if t == pad_id:
            continue
        out.append(t)
    return out


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple[EncoderDecoder, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = EncoderDecoder(
        vocab_size=int(cfg["vocab_size"]),
        pad_id=int(cfg["pad_id"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        n_enc_layers=int(cfg["n_enc_layers"]),
        n_dec_layers=int(cfg["n_dec_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        pe_type=str(cfg["pe_type"]),
        dropout=float(cfg.get("dropout", 0.0)),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[eval] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, cfg


def _detok(tokenizer, token_lists: list[list[int]]) -> list[str]:
    return tokenizer.batch_decode(token_lists, skip_special_tokens=True)


def greedy_translate(
    model: EncoderDecoder,
    tokenizer,
    sources: list[str],
    device: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_seq_len: int,
    max_new_tokens: int = 128,
    batch_size: int = 32,
) -> list[str]:
    """Tokenize sources in batches; greedy decode; return list of detokenized predictions."""
    hyps: list[str] = []
    for i in range(0, len(sources), batch_size):
        chunk = sources[i : i + batch_size]
        enc = tokenizer(chunk, add_special_tokens=True, truncation=True,
                        max_length=max_seq_len, padding=True, return_tensors="pt")
        src = enc["input_ids"].to(device)
        out = model.generate_greedy(src, bos_id=bos_id, eos_id=eos_id,
                                     max_new_tokens=max_new_tokens)
        # Strip leading BOS column and anything after EOS
        trimmed: list[list[int]] = []
        for row in out.tolist():
            row = row[1:]  # drop BOS
            trimmed.append(_strip_after_eos(row, eos_id, pad_id))
        hyps.extend(_detok(tokenizer, trimmed))
    return hyps


def _bleu(preds: list[str], refs: list[str]) -> float:
    import sacrebleu
    return float(sacrebleu.corpus_bleu(preds, [refs]).score)


def _chrf(preds: list[str], refs: list[str]) -> float:
    import sacrebleu
    return float(sacrebleu.corpus_chrf(preds, [refs]).score)


def _ter(preds: list[str], refs: list[str]) -> float:
    import sacrebleu
    return float(sacrebleu.corpus_ter(preds, [refs]).score)


def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    return {
        "bleu": _bleu(preds, refs),
        "chrf": _chrf(preds, refs),
        "ter": _ter(preds, refs),
        "n": len(preds),
    }


def bleu_by_sentence_length(
    preds: list[str],
    refs: list[str],
    sources: list[str],
    buckets: tuple[tuple[int, int], ...] = ((1, 10), (11, 20), (21, 30), (31, 50), (51, 10_000)),
) -> list[dict]:
    """Bucket by source word count; compute BLEU per bucket."""
    out: list[dict] = []
    src_lens = [len(s.split()) for s in sources]
    for lo, hi in buckets:
        idxs = [i for i, L in enumerate(src_lens) if lo <= L <= hi]
        if not idxs:
            out.append({"range": f"{lo}-{hi}", "n": 0, "bleu": None,
                        "chrf": None, "ter": None})
            continue
        p = [preds[i] for i in idxs]
        r = [refs[i] for i in idxs]
        out.append({
            "range": f"{lo}-{hi}",
            "n": len(idxs),
            "bleu": _bleu(p, r),
            "chrf": _chrf(p, r),
            "ter": _ter(p, r),
        })
    return out


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_tsv: str,
    output_dir: str,
    device: str,
    tokenizer_name: str = "Helsinki-NLP/opus-mt-en-de",
    max_new_tokens: int = 128,
    batch_size: int = 32,
) -> dict:
    from src.tokenizer_utils import build_mt_tokenizer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_from_checkpoint(checkpoint_path, device)
    tokenizer = build_mt_tokenizer(tokenizer_name)
    bos_id = int(cfg["bos_id"])
    eos_id = int(cfg["eos_id"])
    pad_id = int(cfg["pad_id"])
    max_seq_len = int(cfg["max_seq_len"])

    pairs = _load_pairs(Path(eval_tsv))
    print(f"[eval] loaded {len(pairs):,} pairs from {eval_tsv}")
    sources = [s for s, _ in pairs]
    refs = [t for _, t in pairs]

    print(f"[eval] greedy decoding (max_new_tokens={max_new_tokens}, batch={batch_size})")
    import time
    t0 = time.monotonic()
    preds = greedy_translate(
        model, tokenizer, sources, device,
        bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        max_seq_len=max_seq_len, max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    decode_sec = time.monotonic() - t0
    print(f"[eval] decoded {len(preds)} in {decode_sec:.1f}s "
          f"({len(preds) / max(decode_sec, 1e-6):.2f} sent/s)")

    overall = compute_metrics(preds, refs)
    by_len = bleu_by_sentence_length(preds, refs, sources)

    # Persist predictions for inspection
    with open(out_dir / "predictions.tsv", "w", encoding="utf-8") as f:
        for s, r, p in zip(sources, refs, preds):
            f.write(f"{s}\t{r}\t{p}\n")

    result = {
        "checkpoint": str(checkpoint_path),
        "eval_tsv": str(eval_tsv),
        "pe_type": cfg.get("pe_type"),
        "n_params": cfg.get("n_params"),
        "decode_sec": decode_sec,
        "overall": overall,
        "by_src_length": by_len,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(result, indent=2))
    print(f"[eval] wrote {out_dir / 'eval_summary.json'}")
    return result
