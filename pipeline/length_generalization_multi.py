"""Length generalization experiment: compare ALL methods side by side.

Extends the original 2-way comparison to support arbitrary number of checkpoints.

Usage:
    python -m pipeline.length_generalization_multi \
        --ckpts "outputs/checkpoints/rope_de/best.pt:RoPE" \
        --ckpts "outputs/checkpoints/asrope3_de/best.pt:AdaptiveRoPE" \
        --ckpts "outputs/checkpoints/gatesonly_de_s42/best.pt:GatesOnly" \
        --ckpts "outputs/checkpoints/phasesonly_de_s42/best.pt:PhasesOnly" \
        --eval-tsv raw_data/wmt14/test.tsv \
        --train-tsv raw_data/wmt14/train.tsv \
        --extend-to 420 \
        --out-dir outputs/analysis/length_gen
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

from src.eval import load_model_from_checkpoint
from src.model import EncoderDecoder
from src.positional import AdaptiveRoPE, GatesOnlyAdaptiveRoPE, PhasesOnlyAdaptiveRoPE, RoPE, ALiBi, Sinusoidal, apply_position_interpolation


def extend_pe_cache(model: EncoderDecoder, new_max_len: int) -> None:
    """Extend positional encoding buffers to new_max_len positions."""
    for module in model.modules():
        if isinstance(module, (AdaptiveRoPE, GatesOnlyAdaptiveRoPE, PhasesOnlyAdaptiveRoPE)):
            device = module.positions.device
            module.positions = torch.arange(new_max_len, dtype=torch.float32, device=device)
        elif isinstance(module, RoPE):
            device = module.cos_cache.device
            head_dim = module.cos_cache.shape[-1] * 2
            inv_freq = 1.0 / (10000.0 ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            positions = torch.arange(new_max_len, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            module.cos_cache = freqs.cos().to(device)
            module.sin_cache = freqs.sin().to(device)
        elif isinstance(module, ALiBi):
            positions = torch.arange(new_max_len)
            distance = positions.unsqueeze(0) - positions.unsqueeze(1)
            distance = distance.abs().unsqueeze(0)
            module.distance = distance.to(module.distance.device)
        elif isinstance(module, Sinusoidal):
            import math
            d_model = module.pe.shape[1]
            pe = torch.zeros(new_max_len, d_model)
            pos = torch.arange(new_max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            module.pe = pe.to(module.pe.device)
    model.max_seq_len = new_max_len


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
    tokenizer,
    sources: list[str],
    device: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_src_len: int,
    max_new_tokens: int = 150,
    batch_size: int = 16,
) -> list[str]:
    from transformers import PreTrainedTokenizerFast
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


LENGTH_BUCKETS = [
    (1,   20,  "1-20"),
    (21,  40,  "21-40"),
    (41,  60,  "41-60"),
    (61,  80,  "61-80"),
    (81,  100, "81-100"),
    (101, 150, "101-150"),
    (151, 200, "151-200"),
    (201, 400, "201-400"),
]
OOD_THRESHOLD_WORDS = 80


def run_experiment(
    model: EncoderDecoder,
    tokenizer,
    pairs: list[tuple[str, str]],
    device: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    label: str,
    max_per_bucket: int = 500,
    rng_seed: int = 42,
) -> list[dict]:
    import random
    rng = random.Random(rng_seed)
    src_word_lens = [len(s.split()) for s, _ in pairs]
    results = []

    for lo, hi, name in LENGTH_BUCKETS:
        idxs = [i for i, L in enumerate(src_word_lens) if lo <= L <= hi]
        if len(idxs) > max_per_bucket:
            idxs = rng.sample(idxs, max_per_bucket)
        bucket_sources = [pairs[i][0] for i in idxs]
        bucket_refs = [pairs[i][1] for i in idxs]
        n = len(idxs)

        if n < 10:
            results.append({"range": name, "lo": lo, "hi": hi,
                            "n": n, "bleu": None, "chrf": None,
                            "ood": lo >= OOD_THRESHOLD_WORDS})
            continue

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


def compute_degradation(bucket_results: list[dict]) -> list[dict]:
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


def plot_results(all_results: dict[str, list[dict]], out_dir: Path) -> None:
    def _extract(results, key):
        xs, ys = [], []
        for r in results:
            if r[key] is not None and r["n"] >= 5:
                mid = (r["lo"] + r["hi"]) / 2
                xs.append(mid)
                ys.append(r[key])
        return np.array(xs), np.array(ys)

    methods = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # BLEU by length
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in zip(methods, colors):
        xr, yr = _extract(all_results[method], "bleu")
        ax.plot(xr, yr, "o-", label=method, linewidth=2, markersize=6, color=color)
    ax.axvline(128, color="gray", linestyle="--", linewidth=1.2, label="Train max_seq_len=128")
    ax.set_xlabel("Source sentence length (words)")
    ax.set_ylabel("BLEU")
    ax.set_title("Length Generalization: BLEU vs Source Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "length_generalization_bleu.png", dpi=150)
    plt.close(fig)

    # BLEU degradation
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in zip(methods, colors):
        xr, yr = _extract(all_results[method], "bleu_drop_pct")
        ax.plot(xr, yr, "s-", label=method, linewidth=2, markersize=6, color=color)
    ax.axvline(128, color="gray", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Source sentence length (words)")
    ax.set_ylabel("BLEU change vs 1-20 baseline (%)")
    ax.set_title("BLEU Degradation Rate (lower drop = better extrapolation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "length_generalization_degradation.png", dpi=150)
    plt.close(fig)

    # chrF by length
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in zip(methods, colors):
        xr, yr = _extract(all_results[method], "chrf")
        ax.plot(xr, yr, "o-", label=method, linewidth=2, markersize=6, color=color)
    ax.axvline(128, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Source sentence length (words)")
    ax.set_ylabel("chrF")
    ax.set_title("Length Generalization: chrF vs Source Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "length_generalization_chrf.png", dpi=150)
    plt.close(fig)


def print_summary_table(all_results: dict[str, list[dict]]) -> None:
    methods = list(all_results.keys())
    print("\n" + "=" * (20 + len(methods) * 18))
    header = f"{'Range':>10}"
    for m in methods:
        header += f" {m[:14]:>14}"
    print(header)
    print("=" * (20 + len(methods) * 18))

    n_buckets = len(next(iter(all_results.values())))
    for i in range(n_buckets):
        row_range = all_results[methods[0]][i]["range"]
        row = f"{row_range:>10}"
        for m in methods:
            b = all_results[m][i]
            val = f"{b['bleu']:.2f}" if b["bleu"] is not None else "—"
            row += f" {val:>14}"
        print(row)
    print("=" * (20 + len(methods) * 18))

    # OOD average drop
    print("\nOOD Average BLEU Drop:")
    for m in methods:
        drops = [r["bleu_drop_pct"] for r in all_results[m]
                 if r.get("ood") and r["bleu_drop_pct"] is not None]
        if drops:
            print(f"  {m:<20} {np.mean(drops):+.2f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Length generalization: multi-method comparison")
    parser.add_argument("--ckpts", action="append", required=True,
                        help="Checkpoint path with label: path/to/best.pt:MethodName")
    parser.add_argument("--eval-tsv", required=True)
    parser.add_argument("--train-tsv", default=None)
    parser.add_argument("--extend-to", type=int, default=420)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-bucket", type=int, default=500)
    parser.add_argument("--out-dir", default="outputs/analysis/length_gen_multi")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse checkpoints
    ckpt_entries = []
    for entry in args.ckpts:
        parts = entry.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"--ckpts must be 'path:label', got: {entry}")
        ckpt_entries.append((parts[0], parts[1]))

    # Load pairs
    def _load_tsv(path: str) -> list[tuple[str, str]]:
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    result.append((parts[0], parts[1]))
        return result

    pairs = _load_tsv(args.eval_tsv)
    print(f"[length_gen] loaded {len(pairs):,} test pairs")

    if args.train_tsv:
        train_pairs = _load_tsv(args.train_tsv)
        long_pairs = [(s, t) for s, t in train_pairs if len(s.split()) >= OOD_THRESHOLD_WORDS]
        print(f"[length_gen] adding {len(long_pairs):,} long train sentences for OOD buckets")
        pairs = pairs + long_pairs

    src_lens = [len(s.split()) for s, _ in pairs]
    print(f"[length_gen] combined source length: mean={np.mean(src_lens):.1f} max={max(src_lens)} p95={np.percentile(src_lens, 95):.0f}")

    # Run evaluation for each checkpoint
    all_results = {}
    for ckpt_path, label in ckpt_entries:
        print(f"\n[length_gen] Loading {label}: {ckpt_path}")
        model, cfg = load_model_from_checkpoint(ckpt_path, device)
        extend_pe_cache(model, args.extend_to)

        from src.tokenizer_utils import build_mt_tokenizer
        tokenizer_name = cfg.get("tokenizer", "Helsinki-NLP/opus-mt-en-de")
        tokenizer = build_mt_tokenizer(tokenizer_name)

        bos_id = int(cfg["bos_id"])
        eos_id = int(cfg["eos_id"])
        pad_id = int(cfg["pad_id"])

        print(f"[length_gen] Evaluating {label} across length buckets:")
        results = run_experiment(
            model, tokenizer, pairs, device,
            bos_id, eos_id, pad_id, label=label,
            max_per_bucket=args.max_per_bucket,
        )
        results = compute_degradation(results)
        all_results[label] = results

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print_summary_table(all_results)

    # Plots
    plot_results(all_results, out_dir)

    # Save JSON
    output = {
        "extend_to": args.extend_to,
        "n_pairs_total": len(pairs),
        "methods": {label: {"path": path, "results": all_results[label]}
                    for path, label in ckpt_entries},
    }
    (out_dir / "length_gen_multi_results.json").write_text(json.dumps(output, indent=2))
    print(f"\n[length_gen] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
