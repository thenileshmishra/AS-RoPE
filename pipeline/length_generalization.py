"""Length generalization experiment: RoPE vs Adaptive RoPE on WMT14 En-De.

Tests both models on sentences stratified by length, including out-of-distribution
lengths beyond the training max_seq_len (256). Measures BLEU/chrF degradation rate
to show whether Adaptive RoPE's learned frequency structure extrapolates better.

Core hypothesis: Adaptive RoPE (learned gates < 1) biases toward low-frequency
positional structure → smoother degradation on unseen long sequences.

Usage:
    python -m pipeline.length_generalization \
        --rope-ckpt     outputs/checkpoints/rope_de/best.pt \
        --adaptive-ckpt outputs/checkpoints/asrope3_de/best.pt \
        --eval-tsv      raw_data/wmt14/test.tsv \
        --extend-to     400 \
        --out-dir       outputs/analysis/length_gen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from src.eval import load_model_from_checkpoint
from src.model import EncoderDecoder
from src.positional import AdaptiveRoPE, RoPE


# ---------------------------------------------------------------------------
# Cache extension (no retraining — purely inference-time)
# ---------------------------------------------------------------------------

def extend_pe_cache(model: EncoderDecoder, new_max_len: int) -> None:
    """Extend positional encoding buffers to new_max_len positions.

    For RoPE:         recompute cos/sin cache with more positions.
    For AdaptiveRoPE: extend the base positions tensor; learned gates/phases
                      are applied per-position so they naturally extrapolate.
    No gradient needed — purely inference manipulation.
    """
    for module in model.modules():
        if isinstance(module, AdaptiveRoPE):
            device = module.positions.device
            module.positions = torch.arange(new_max_len, dtype=torch.float32,
                                            device=device)
        elif isinstance(module, RoPE):
            device = module.cos_cache.device
            # Rebuild cache with same inv_freq, more positions
            head_dim = module.cos_cache.shape[-1] * 2
            inv_freq = 1.0 / (10000.0 ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            positions = torch.arange(new_max_len, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            module.cos_cache = freqs.cos().to(device)
            module.sin_cache = freqs.sin().to(device)

    # Update model's own max_seq_len so tokenizer/generate limits work
    model.max_seq_len = new_max_len


# ---------------------------------------------------------------------------
# Bucketed translation (does NOT truncate source beyond bucket upper bound)
# ---------------------------------------------------------------------------

def _strip_after_eos(ids: list[int], eos_id: int, pad_id: int) -> list[int]:
    out = []
    for t in ids:
        if t == eos_id:
            break
        if t != pad_id:
            out.append(t)
    return out


def translate_bucket(
    model: EncoderDecoder,
    tokenizer: PreTrainedTokenizerFast,
    sources: list[str],
    device: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_src_len: int,      # true limit for this bucket (no truncation below this)
    max_new_tokens: int = 150,
    batch_size: int = 16,
) -> list[str]:
    """Greedy translate without truncating sentences shorter than max_src_len."""
    hyps: list[str] = []
    for i in range(0, len(sources), batch_size):
        chunk = sources[i: i + batch_size]
        enc = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=True,
            max_length=max_src_len,
            padding=True,
            return_tensors="pt",
        )
        src = enc["input_ids"].to(device)
        out = model.generate_greedy(src, bos_id=bos_id, eos_id=eos_id,
                                     max_new_tokens=max_new_tokens)
        for row in out.tolist():
            row = row[1:]  # drop leading BOS
            hyps.append(
                tokenizer.decode(_strip_after_eos(row, eos_id, pad_id),
                                 skip_special_tokens=True)
            )
    return hyps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bleu(preds: list[str], refs: list[str]) -> float | None:
    if not preds:
        return None
    import sacrebleu
    return float(sacrebleu.corpus_bleu(preds, [refs]).score)


def _chrf(preds: list[str], refs: list[str]) -> float | None:
    if not preds:
        return None
    import sacrebleu
    return float(sacrebleu.corpus_chrf(preds, [refs]).score)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

LENGTH_BUCKETS = [
    (1,   20,  "1-20"),
    (21,  40,  "21-40"),
    (41,  60,  "41-60"),
    (61,  80,  "61-80"),     # ~90-120 tokens after tokenization
    (81,  100, "81-100"),    # ~120-150 tokens
    (101, 150, "101-150"),   # ~150-225 tokens — boundary zone
    (151, 200, "151-200"),   # ~225-300 tokens — clearly OOD
    (201, 400, "201-400"),   # OOD (deep)
]
OOD_THRESHOLD_WORDS = 80    # sentences longer than this are OOD (tokens > ~120)


def run_experiment(
    model: EncoderDecoder,
    tokenizer: PreTrainedTokenizerFast,
    pairs: list[tuple[str, str]],
    device: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    label: str,
    max_per_bucket: int = 500,
    rng_seed: int = 42,
) -> list[dict]:
    """Evaluate model across all length buckets. Returns per-bucket metrics."""
    import random
    rng = random.Random(rng_seed)

    src_word_lens = [len(s.split()) for s, _ in pairs]
    results = []

    for lo, hi, name in LENGTH_BUCKETS:
        idxs = [i for i, L in enumerate(src_word_lens) if lo <= L <= hi]
        if len(idxs) > max_per_bucket:
            idxs = rng.sample(idxs, max_per_bucket)
        bucket_sources = [pairs[i][0] for i in idxs]
        bucket_refs    = [pairs[i][1] for i in idxs]
        n = len(idxs)

        if n < 10:
            print(f"  [{label}] bucket {name:>8}  n={n:>4}  (skipped — too few)")
            results.append({"range": name, "lo": lo, "hi": hi,
                            "n": n, "bleu": None, "chrf": None,
                            "ood": lo >= OOD_THRESHOLD_WORDS})
            continue

        # For OOD buckets: allow sequences to use the extended cache
        # For in-dist: cap at model's trained max_seq_len
        bucket_max = model.max_seq_len if lo < OOD_THRESHOLD_WORDS else model.max_seq_len
        t0 = time.monotonic()
        preds = translate_bucket(
            model, tokenizer, bucket_sources, device,
            bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            max_src_len=bucket_max,
        )
        elapsed = time.monotonic() - t0

        bleu = _bleu(preds, bucket_refs)
        chrf = _chrf(preds, bucket_refs)
        ood_flag = "OOD" if lo >= OOD_THRESHOLD_WORDS else "   "

        print(f"  [{label}] bucket {name:>8}  n={n:>4}  "
              f"BLEU={bleu:>6.2f}  chrF={chrf:>6.2f}  "
              f"{ood_flag}  ({elapsed:.1f}s)")

        results.append({
            "range": name, "lo": lo, "hi": hi,
            "n": n, "bleu": bleu, "chrf": chrf,
            "ood": lo >= OOD_THRESHOLD_WORDS,
        })

    return results


# ---------------------------------------------------------------------------
# Degradation analysis
# ---------------------------------------------------------------------------

def compute_degradation(bucket_results: list[dict]) -> list[dict]:
    """Compute degradation relative to the [1-20] baseline bucket."""
    baseline = next((r for r in bucket_results if r["range"] == "1-20"), None)
    if baseline is None or baseline["bleu"] is None:
        return bucket_results

    base_bleu = baseline["bleu"]
    base_chrf = baseline["chrf"]

    for r in bucket_results:
        if r["bleu"] is not None:
            r["bleu_drop_pct"] = (r["bleu"] - base_bleu) / base_bleu * 100
            r["chrf_drop_pct"] = (r["chrf"] - base_chrf) / base_chrf * 100
        else:
            r["bleu_drop_pct"] = None
            r["chrf_drop_pct"] = None
    return bucket_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    rope_results: list[dict],
    adaptive_results: list[dict],
    out_dir: Path,
) -> None:
    def _extract(results, key):
        xs, ys = [], []
        for r in results:
            if r[key] is not None and r["n"] >= 5:
                mid = (r["lo"] + r["hi"]) / 2
                xs.append(mid)
                ys.append(r[key])
        return np.array(xs), np.array(ys)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Length Generalization: RoPE vs Adaptive RoPE (WMT14 En-De)",
                 fontsize=13, fontweight="bold")

    # ---- BLEU by length ----
    ax = axes[0]
    xr, yr = _extract(rope_results, "bleu")
    xa, ya = _extract(adaptive_results, "bleu")
    ax.plot(xr, yr, "o-", color="#1f77b4", label="RoPE", linewidth=2, markersize=6)
    ax.plot(xa, ya, "s-", color="#d62728", label="Adaptive RoPE", linewidth=2, markersize=6)
    ax.axvline(256, color="gray", linestyle="--", linewidth=1.2, label="Train max_seq_len=256")
    ax.set_xlabel("Source sentence length (words)")
    ax.set_ylabel("BLEU")
    ax.set_title("BLEU vs Source Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- BLEU drop % relative to short sentences ----
    ax = axes[1]
    xr, yr = _extract(rope_results, "bleu_drop_pct")
    xa, ya = _extract(adaptive_results, "bleu_drop_pct")
    ax.plot(xr, yr, "o-", color="#1f77b4", label="RoPE", linewidth=2, markersize=6)
    ax.plot(xa, ya, "s-", color="#d62728", label="Adaptive RoPE", linewidth=2, markersize=6)
    ax.axvline(256, color="gray", linestyle="--", linewidth=1.2, label="Train max_seq_len=256")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Source sentence length (words)")
    ax.set_ylabel("BLEU change vs 1-30 baseline (%)")
    ax.set_title("BLEU Degradation Rate (lower drop = better extrapolation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "length_generalization.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[length_gen] plot saved: {out_path}")

    # ---- chrF plot ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    xr, yr = _extract(rope_results, "chrf")
    xa, ya = _extract(adaptive_results, "chrf")
    ax2.plot(xr, yr, "o-", color="#1f77b4", label="RoPE", linewidth=2, markersize=6)
    ax2.plot(xa, ya, "s-", color="#d62728", label="Adaptive RoPE", linewidth=2, markersize=6)
    ax2.axvline(256, color="gray", linestyle="--", linewidth=1.2, label="Train max_seq_len=256")
    ax2.set_xlabel("Source sentence length (words)")
    ax2.set_ylabel("chrF")
    ax2.set_title("chrF vs Source Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "length_generalization_chrf.png", dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(rope: list[dict], adaptive: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"{'Range':>10} {'n_rope':>7} {'BLEU_rope':>10} {'BLEU_adap':>10} "
          f"{'diff':>7} {'drop_rope%':>11} {'drop_adap%':>11}")
    print("=" * 80)
    ood_rope_drops = []
    ood_adap_drops = []
    for r, a in zip(rope, adaptive):
        n = r["n"]
        b_r = f"{r['bleu']:.2f}" if r["bleu"] is not None else "—"
        b_a = f"{a['bleu']:.2f}" if a["bleu"] is not None else "—"
        diff = ""
        if r["bleu"] is not None and a["bleu"] is not None:
            d = a["bleu"] - r["bleu"]
            diff = f"{d:+.2f}"
        dr = f"{r.get('bleu_drop_pct', None):.1f}%" if r.get("bleu_drop_pct") is not None else "—"
        da = f"{a.get('bleu_drop_pct', None):.1f}%" if a.get("bleu_drop_pct") is not None else "—"

        ood = "  ← OOD" if r.get("ood") else ""
        print(f"{r['range']:>10} {n:>7} {b_r:>10} {b_a:>10} {diff:>7} {dr:>11} {da:>11}{ood}")

        if r.get("ood") and r["bleu"] is not None and a["bleu"] is not None:
            if r.get("bleu_drop_pct") is not None:
                ood_rope_drops.append(r["bleu_drop_pct"])
            if a.get("bleu_drop_pct") is not None:
                ood_adap_drops.append(a["bleu_drop_pct"])

    print("=" * 80)
    if ood_rope_drops and ood_adap_drops:
        avg_rope = np.mean(ood_rope_drops)
        avg_adap = np.mean(ood_adap_drops)
        print(f"\nOOD average BLEU drop:  RoPE = {avg_rope:.1f}%  |  Adaptive RoPE = {avg_adap:.1f}%")
        diff_drop = avg_adap - avg_rope
        if diff_drop < 0:
            print(f"→ Adaptive RoPE degrades {abs(diff_drop):.1f}% LESS than RoPE on OOD lengths.")
        else:
            print(f"→ Adaptive RoPE degrades {diff_drop:.1f}% MORE than RoPE on OOD lengths.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Length generalization: RoPE vs Adaptive RoPE")
    parser.add_argument("--rope-ckpt",     required=True,
                        help="Path to standard RoPE best.pt")
    parser.add_argument("--adaptive-ckpt", required=True,
                        help="Path to Adaptive RoPE best.pt")
    parser.add_argument("--eval-tsv",      required=True,
                        help="Test TSV (short sentences)")
    parser.add_argument("--train-tsv",     default=None,
                        help="Train TSV — used to sample long sentences for OOD buckets")
    parser.add_argument("--tokenizer",     default="Helsinki-NLP/opus-mt-en-de")
    parser.add_argument("--extend-to",     type=int, default=420,
                        help="Extend PE cache to this many positions (>= 256 for OOD)")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--batch-size",    type=int, default=16)
    parser.add_argument("--max-per-bucket", type=int, default=500,
                        help="Max sentences to sample per length bucket")
    parser.add_argument("--out-dir",       default="outputs/analysis/length_gen")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[length_gen] device={device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load_tsv(path: str) -> list[tuple[str, str]]:
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    result.append((parts[0], parts[1]))
        return result

    # ---- Load test set ----
    pairs = _load_tsv(args.eval_tsv)
    print(f"[length_gen] loaded {len(pairs):,} test pairs")

    # ---- Augment with training sentences for long buckets ----
    if args.train_tsv:
        print(f"[length_gen] loading train TSV for long-sentence OOD buckets...")
        train_pairs = _load_tsv(args.train_tsv)
        # Keep only long sentences (OOD territory) to avoid contaminating short buckets
        long_pairs = [(s, t) for s, t in train_pairs if len(s.split()) >= OOD_THRESHOLD_WORDS]
        print(f"[length_gen] train pairs with >={OOD_THRESHOLD_WORDS} words: {len(long_pairs):,}")
        pairs = pairs + long_pairs

    # ---- Print length distribution ----
    src_lens = [len(s.split()) for s, _ in pairs]
    print(f"[length_gen] combined source length: mean={np.mean(src_lens):.1f} "
          f"max={max(src_lens)} p95={np.percentile(src_lens, 95):.0f}")

    # ---- Load tokenizer ----
    from src.tokenizer_utils import build_mt_tokenizer
    tokenizer = build_mt_tokenizer(args.tokenizer)

    # ---- Load and extend RoPE model ----
    print("\n[length_gen] Loading RoPE checkpoint...")
    rope_model, rope_cfg = load_model_from_checkpoint(args.rope_ckpt, device)
    extend_pe_cache(rope_model, args.extend_to)
    bos_id = int(rope_cfg["bos_id"])
    eos_id = int(rope_cfg["eos_id"])
    pad_id = int(rope_cfg["pad_id"])
    print(f"[length_gen] RoPE PE cache extended to {args.extend_to}")

    print("\n[length_gen] Evaluating RoPE across length buckets:")
    rope_results = run_experiment(
        rope_model, tokenizer, pairs, device,
        bos_id, eos_id, pad_id, label="RoPE",
        max_per_bucket=args.max_per_bucket,
    )
    rope_results = compute_degradation(rope_results)
    del rope_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Load and extend Adaptive RoPE model ----
    print("\n[length_gen] Loading Adaptive RoPE checkpoint...")
    adap_model, adap_cfg = load_model_from_checkpoint(args.adaptive_ckpt, device)
    extend_pe_cache(adap_model, args.extend_to)
    print(f"[length_gen] Adaptive RoPE PE cache extended to {args.extend_to}")

    print("\n[length_gen] Evaluating Adaptive RoPE across length buckets:")
    adap_results = run_experiment(
        adap_model, tokenizer, pairs, device,
        bos_id, eos_id, pad_id, label="AdaptiveRoPE",
        max_per_bucket=args.max_per_bucket,
    )
    adap_results = compute_degradation(adap_results)
    del adap_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Summary ----
    print_summary_table(rope_results, adap_results)

    # ---- Plots ----
    plot_results(rope_results, adap_results, out_dir)

    # ---- Save full results JSON ----
    output = {
        "rope_checkpoint": args.rope_ckpt,
        "adaptive_checkpoint": args.adaptive_ckpt,
        "eval_tsv": args.eval_tsv,
        "extend_to": args.extend_to,
        "n_pairs_total": len(pairs),
        "rope": rope_results,
        "adaptive": adap_results,
    }
    json_path = out_dir / "length_gen_results.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"[length_gen] full results JSON: {json_path}")


if __name__ == "__main__":
    main()
