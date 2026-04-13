"""Decoding (greedy + beam) and comprehensive MT evaluation.

Metrics: BLEU, chrF, TER, BERTScore, COMET, length ratio, repetition rate.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from src.dataset import load_pairs
from src.model import GPT
from src.tokenizer_utils import build_mt_tokenizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

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
        pe_type=str(config.get("pe_type", config.get("positional_encoding", "sinusoidal"))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Greedy decoding (original)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode_one(
    model: GPT,
    tokenizer,
    src_text: str,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 4,
) -> str:
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
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
        next_logits = logits[0, -1].clone()

        # Repetition penalty: downscale logits for tokens already generated
        if repetition_penalty != 1.0 and generated:
            seen = set(generated)
            for tid in seen:
                if next_logits[tid] > 0:
                    next_logits[tid] /= repetition_penalty
                else:
                    next_logits[tid] *= repetition_penalty

        # No-repeat n-gram blocking
        if no_repeat_ngram_size > 1 and len(generated) >= no_repeat_ngram_size - 1:
            ngram_prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
            for i in range(len(generated) - no_repeat_ngram_size + 1):
                if tuple(generated[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    blocked = generated[i + no_repeat_ngram_size - 1]
                    next_logits[blocked] = float("-inf")

        next_id = int(next_logits.argmax().item())
        if next_id == eos_id:
            break
        generated.append(next_id)
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    return " ".join(tokenizer.decode(generated, skip_special_tokens=True).split()).strip()


# ---------------------------------------------------------------------------
# Beam search decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_decode_one(
    model: GPT,
    tokenizer,
    src_text: str,
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
    beam_size: int = 5,
    length_penalty: float = 0.6,
) -> str:
    """Beam search decoding for a single source sentence."""
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    src_ids = tokenizer(src_text, add_special_tokens=False)["input_ids"]
    max_prompt = max(1, max_seq_len - max_new_tokens - 1)
    if len(src_ids) > max_prompt:
        src_ids = src_ids[:max_prompt]

    prompt = src_ids + [sep_id]

    # Each beam: (cumulative_log_prob, list_of_token_ids)
    beams: list[tuple[float, list[int]]] = [(0.0, prompt)]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_new_tokens):
        if not beams:
            break

        candidates: list[tuple[float, list[int]]] = []
        for log_prob, tokens in beams:
            if len(tokens) >= max_seq_len:
                completed.append((log_prob, tokens))
                continue
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _ = model(input_ids)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            topk = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                tid = int(topk.indices[i].item())
                lp = float(topk.values[i].item())
                new_tokens = tokens + [tid]
                new_log_prob = log_prob + lp
                if tid == eos_id:
                    completed.append((new_log_prob, new_tokens))
                else:
                    candidates.append((new_log_prob, new_tokens))

        if not candidates and not completed:
            break

        # Score and prune candidates
        def _score(lp: float, length: int) -> float:
            return lp / (length ** length_penalty)

        candidates.sort(key=lambda x: _score(x[0], len(x[1])), reverse=True)
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            # Early stop: best completed beam scores higher than best active beam
            best_completed = max(completed, key=lambda x: _score(x[0], len(x[1])))
            best_active = beams[0] if beams else None
            if best_active is None or _score(best_completed[0], len(best_completed[1])) >= _score(best_active[0], len(best_active[1])):
                break

    # If no beam completed, use the best active beam
    if not completed:
        completed = beams

    if not completed:
        return ""

    def _final_score(lp: float, length: int) -> float:
        return lp / (length ** length_penalty)

    best = max(completed, key=lambda x: _final_score(x[0], len(x[1])))
    # Strip prompt tokens from the output
    gen_ids = best[1][len(prompt):]
    # Remove trailing eos if present
    if gen_ids and gen_ids[-1] == eos_id:
        gen_ids = gen_ids[:-1]

    return " ".join(tokenizer.decode(gen_ids, skip_special_tokens=True).split()).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeated in the given text."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def compute_all_metrics(
    preds: list[str],
    refs: list[str],
    sources: list[str] | None = None,
    metrics_mode: str = "full",
) -> dict:
    """Compute MT metrics.

    metrics_mode="fast": BLEU + chrF + TER + length_ratio + repetition_rate only.
    metrics_mode="full": also runs BERTScore and COMET (slow, downloads models).
    """
    if len(preds) != len(refs):
        raise ValueError(f"Pred/ref count mismatch: {len(preds)} vs {len(refs)}")
    if not preds:
        raise ValueError("Empty prediction list")
    if metrics_mode not in ("fast", "full"):
        raise ValueError(f"metrics_mode must be 'fast' or 'full', got {metrics_mode!r}")

    metrics: dict = {"num_samples": len(preds), "metrics_mode": metrics_mode}

    # --- sacrebleu metrics (always on, fast) ---
    from sacrebleu.metrics import BLEU, CHRF, TER

    bleu_m = BLEU()
    chrf_m = CHRF()
    ter_m = TER()

    bleu_score = bleu_m.corpus_score(preds, [refs])
    chrf_score = chrf_m.corpus_score(preds, [refs])
    ter_score = ter_m.corpus_score(preds, [refs])

    metrics["bleu"] = float(bleu_score.score)
    metrics["chrf"] = float(chrf_score.score)
    metrics["ter"] = float(ter_score.score)
    metrics["bleu_signature"] = str(bleu_m.get_signature())
    metrics["chrf_signature"] = str(chrf_m.get_signature())
    metrics["ter_signature"] = str(ter_m.get_signature())

    metrics["bertscore_f1"] = None
    metrics["comet"] = None

    if metrics_mode == "full":
        # --- BERTScore ---
        try:
            from bert_score import score as bs_score

            _, _, f1 = bs_score(preds, refs, lang="en", verbose=False)
            metrics["bertscore_f1"] = float(f1.mean().item())
        except ImportError:
            warnings.warn("bert-score not installed; skipping BERTScore")
        except Exception as e:
            warnings.warn(f"BERTScore failed: {e}")

        # --- COMET ---
        if sources is not None:
            try:
                from comet import download_model, load_from_checkpoint

                model_path = download_model("Unbabel/wmt22-comet-da")
                comet_model = load_from_checkpoint(model_path)
                comet_data = [
                    {"src": s, "mt": p, "ref": r}
                    for s, p, r in zip(sources, preds, refs)
                ]
                comet_output = comet_model.predict(comet_data, batch_size=64, gpus=0)
                metrics["comet"] = float(comet_output.system_score)
            except ImportError:
                warnings.warn("unbabel-comet not installed; skipping COMET")
            except Exception as e:
                warnings.warn(f"COMET failed: {e}")

    # --- Length ratio ---
    ratios = [
        len(p.split()) / max(len(r.split()), 1)
        for p, r in zip(preds, refs)
    ]
    metrics["length_ratio"] = sum(ratios) / len(ratios)

    # --- Repetition rate ---
    rep_rates = [_repetition_rate(p) for p in preds]
    metrics["repetition_rate"] = sum(rep_rates) / len(rep_rates)

    return metrics


def compute_bleu_chrf(preds: list[str], refs: list[str]) -> dict:
    """Backward-compatible wrapper that calls compute_all_metrics in fast mode."""
    return compute_all_metrics(preds, refs, metrics_mode="fast")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _format_summary(metrics: dict, decoding_label: str = "greedy") -> str:
    lines = [
        "═══════════════════════════════════",
        "Evaluation Results",
        "═══════════════════════════════════",
        f"{'Decoding':<20}{decoding_label}",
        f"{'Num Samples':<20}{metrics.get('num_samples', '?')}",
        "───────────────────────────────────",
    ]
    for key, fmt in [
        ("bleu", ".2f"),
        ("chrf", ".2f"),
        ("ter", ".2f"),
        ("bertscore_f1", ".3f"),
        ("comet", ".3f"),
        ("length_ratio", ".2f"),
        ("repetition_rate", ".3f"),
    ]:
        val = metrics.get(key)
        label = key.replace("_", " ").replace("bertscore f1", "BERTScore-F1").replace("bleu", "BLEU").replace("chrf", "chrF").replace("ter", "TER").replace("comet", "COMET").replace("length ratio", "Length Ratio").replace("repetition rate", "Repetition Rate")
        if val is not None:
            lines.append(f"{label:<20}{val:{fmt}}")
        else:
            lines.append(f"{label:<20}N/A")
    lines.append("═══════════════════════════════════")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation entry points
# ---------------------------------------------------------------------------

def evaluate_checkpoint_v2(
    checkpoint_path: str | Path,
    eval_tsv: str | Path,
    output_dir: str | Path,
    max_new_tokens: int = 128,
    device: str = "cpu",
    decoding: str = "beam",
    beam_size: int = 5,
    metrics_mode: str = "full",
) -> dict:
    """Decode + evaluate with comprehensive metrics. Saves all artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_checkpoint(checkpoint_path, device)
    tok_name = str(config.get("tokenizer", "ai4bharat/IndicBART"))
    tokenizer = build_mt_tokenizer(tok_name)
    # Resize embeddings if tokenizer has extra tokens (e.g. <sep>)
    if len(tokenizer) > model.vocab_size:
        model.token_emb = torch.nn.Embedding(len(tokenizer), model.d_model).to(device)
        model.lm_head = torch.nn.Linear(model.d_model, len(tokenizer), bias=False).to(device)
        model.lm_head.weight = model.token_emb.weight
        model.vocab_size = len(tokenizer)
        # Reload weights that still match
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for name, param in ckpt["model_state_dict"].items():
            if name in dict(model.named_parameters()):
                target = dict(model.named_parameters())[name]
                if param.shape == target.shape:
                    target.data.copy_(param)

    max_seq_len = int(config["max_seq_len"])

    pairs = load_pairs(eval_tsv)
    if not pairs:
        raise ValueError(f"No eval pairs loaded from {eval_tsv}")

    decode_fn = greedy_decode_one if decoding == "greedy" else beam_decode_one
    decoding_label = "greedy" if decoding == "greedy" else f"beam (k={beam_size})"

    sources: list[str] = []
    preds: list[str] = []
    refs: list[str] = []
    samples: list[dict] = []

    for i, (src, tgt) in enumerate(pairs):
        if decoding == "greedy":
            pred = greedy_decode_one(model, tokenizer, src, max_new_tokens, max_seq_len, device)
        else:
            pred = beam_decode_one(model, tokenizer, src, max_new_tokens, max_seq_len, device, beam_size=beam_size)
        ref = " ".join(tgt.split()).strip()
        sources.append(src)
        preds.append(pred)
        refs.append(ref)
        samples.append({"src": src, "pred": pred, "ref": ref})
        if (i + 1) % 100 == 0:
            print(f"[eval] decoded {i + 1}/{len(pairs)} ({decoding_label})")

    # Save artifacts
    (output_dir / "pred.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
    (output_dir / "ref.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    (output_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    with open(output_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    metrics = compute_all_metrics(preds, refs, sources=sources, metrics_mode=metrics_mode)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["eval_tsv"] = str(eval_tsv)
    metrics["decoding"] = decoding
    if decoding == "beam":
        metrics["beam_size"] = beam_size

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_txt = _format_summary(metrics, decoding_label)
    (output_dir / "metrics_summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)

    return metrics


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    eval_tsv: str | Path,
    output_dir: str | Path,
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> dict:
    """Backward-compatible wrapper: greedy decoding only."""
    return evaluate_checkpoint_v2(
        checkpoint_path, eval_tsv, output_dir,
        max_new_tokens=max_new_tokens, device=device,
        decoding="greedy",
    )


# ---------------------------------------------------------------------------
# Decoding comparison (greedy vs beam ablation)
# ---------------------------------------------------------------------------

def compare_decoding(
    checkpoint_path: str | Path,
    eval_tsv: str | Path,
    output_dir: str | Path,
    max_new_tokens: int = 128,
    device: str = "cpu",
    beam_sizes: list[int] | None = None,
    metrics_mode: str = "full",
) -> dict:
    """Run greedy (+ optional beam) and produce a comparison table.

    metrics_mode="fast": greedy only, sacrebleu metrics only. Used for the
        3 ablation runs. Overrides beam_sizes.
    metrics_mode="full": greedy + beam_sizes, all metrics including COMET.
        Used once on the winning PE type.
    """
    output_dir = Path(output_dir)
    results: dict[str, dict] = {}

    # Greedy — always run
    greedy_dir = output_dir / "greedy"
    print(f"[compare] running greedy decoding (metrics_mode={metrics_mode}) ...")
    results["greedy"] = evaluate_checkpoint_v2(
        checkpoint_path, eval_tsv, greedy_dir,
        max_new_tokens=max_new_tokens, device=device, decoding="greedy",
        metrics_mode=metrics_mode,
    )

    # Beam variants — skipped entirely in fast mode
    if metrics_mode == "full":
        if beam_sizes is None:
            beam_sizes = [5]
        for bs in beam_sizes:
            if bs == 1:
                continue
            key = f"beam_{bs}"
            beam_dir = output_dir / key
            print(f"[compare] running beam search (k={bs}) ...")
            results[key] = evaluate_checkpoint_v2(
                checkpoint_path, eval_tsv, beam_dir,
                max_new_tokens=max_new_tokens, device=device,
                decoding="beam", beam_size=bs,
                metrics_mode=metrics_mode,
            )

    # Build comparison table
    header_keys = ["bleu", "chrf", "ter", "comet", "bertscore_f1"]
    header_labels = ["BLEU", "chrF", "TER", "COMET", "BERTScore-F1"]
    col_w = 14
    header = f"{'Decoding':<16}" + "".join(f"{lbl:>{col_w}}" for lbl in header_labels)
    sep_line = "─" * len(header)
    rows = [header, sep_line]
    for label, m in results.items():
        row = f"{label:<16}"
        for k in header_keys:
            v = m.get(k)
            if v is not None:
                row += f"{v:>{col_w}.2f}"
            else:
                row += f"{'N/A':>{col_w}}"
        rows.append(row)

    table_txt = "\n".join(rows)
    print("\n" + table_txt)

    (output_dir / "decoding_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "decoding_comparison.txt").write_text(table_txt, encoding="utf-8")

    return results
