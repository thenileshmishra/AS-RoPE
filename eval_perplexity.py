import argparse
import math
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from model import GPT

GPT2_VOCAB_URL = "https://huggingface.co/gpt2/resolve/main/vocab.json"
GPT2_MERGES_URL = "https://huggingface.co/gpt2/resolve/main/merges.txt"
TINY_SHAKESPEARE_URL = "https://huggingface.co/datasets/karpathy/tiny_shakespeare/resolve/main/tiny_shakespeare.txt"


def parse_context_lengths(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths:
        raise ValueError("At least one context length is required.")
    if any(length <= 1 for length in lengths):
        raise ValueError("All context lengths must be > 1.")
    return sorted(set(lengths))


def build_gpt2_tokenizer(cache_dir: str) -> ByteLevelBPETokenizer:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    vocab_path = cache_path / "vocab.json"
    merges_path = cache_path / "merges.txt"
    if not vocab_path.exists():
        urlretrieve(GPT2_VOCAB_URL, vocab_path)
    if not merges_path.exists():
        urlretrieve(GPT2_MERGES_URL, merges_path)

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def load_tiny_shakespeare_tokens(tokenizer: ByteLevelBPETokenizer, cache_dir: str) -> torch.Tensor:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tiny_path = cache_path / "tiny_shakespeare.txt"
    if not tiny_path.exists():
        urlretrieve(TINY_SHAKESPEARE_URL, tiny_path)

    text = tiny_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text).ids
    return torch.tensor(token_ids, dtype=torch.long)


@torch.no_grad()
def perplexity_sliding_no_double_count(
    model: GPT,
    data: torch.Tensor,
    context_len: int,
    device: str,
) -> float:
    model.eval()

    seq_len = data.size(0)
    if seq_len < 2:
        raise ValueError("Not enough tokens to evaluate perplexity.")

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, seq_len - 1, context_len):
        begin = max(i - context_len, 0)
        end = min(i + context_len, seq_len - 1)
        target_len = end - i
        if target_len <= 0:
            continue

        input_ids = data[begin:end].unsqueeze(0).to(device)
        labels = data[begin + 1 : end + 1].unsqueeze(0).to(device)

        if labels.size(1) > target_len:
            labels[:, : labels.size(1) - target_len] = -100

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        total_nll += float(loss.item())
        total_tokens += int(target_len)

    return math.exp(total_nll / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--tokenizer_cache", type=str, default=".cache/gpt2")
    parser.add_argument("--context_lengths", type=str, default="512,1024,2048,4096")
    parser.add_argument("--max_eval_tokens", type=int, default=32768)
    parser.add_argument("--auto_extend_if_no_degradation", action="store_true")
    parser.add_argument("--extend_to", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    context_lengths = parse_context_lengths(args.context_lengths)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    tokenizer = build_gpt2_tokenizer(args.tokenizer_cache)
    data = load_tiny_shakespeare_tokens(tokenizer, args.tokenizer_cache)
    if args.max_eval_tokens > 0 and data.size(0) > args.max_eval_tokens:
        data = data[: args.max_eval_tokens]

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    config = checkpoint.get("config", {})
    vocab_size = int(config.get("vocab_size", tokenizer.get_vocab_size()))

    if int(data.max().item()) >= vocab_size:
        raise ValueError(
            f"Found token id >= vocab_size ({vocab_size}). "
            "Checkpoint must be trained with GPT-2 tokenizer (vocab_size=50257)."
        )

    model = GPT(
        vocab_size=vocab_size,
        d_model=int(config.get("d_model", 256)),
        n_layers=int(config.get("n_layers", 6)),
        n_heads=int(config.get("n_heads", 8)),
        max_seq_len=max(int(config.get("max_seq_len", 1024)), max(context_lengths), args.extend_to),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    per_context: dict[int, float] = {}
    for context_len in context_lengths:
        ppl = perplexity_sliding_no_double_count(
            model=model,
            data=data,
            context_len=context_len,
            device=args.device,
        )
        per_context[context_len] = ppl
        print(f"context={context_len} ppl={ppl:.6f}")

    if 1024 in per_context:
        reference = per_context[1024]
        longer = [length for length in per_context if length > 1024]
        degraded = any(per_context[length] > reference for length in longer)
        print(f"degradation_beyond_1024={degraded}")

        if not degraded and args.auto_extend_if_no_degradation and args.extend_to > max(context_lengths):
            extra_lengths = [length for length in (6144, 8192, 12288, 16384) if max(context_lengths) < length <= args.extend_to]
            for context_len in extra_lengths:
                try:
                    ppl = perplexity_sliding_no_double_count(
                        model=model,
                        data=data,
                        context_len=context_len,
                        device=args.device,
                    )
                    per_context[context_len] = ppl
                    print(f"context={context_len} ppl={ppl:.6f}")
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        print(f"context={context_len} skipped_due_to_oom")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

            longer = [length for length in per_context if length > 1024]
            degraded = any(per_context[length] > reference for length in longer)
            print(f"degradation_beyond_1024_after_extension={degraded}")


if __name__ == "__main__":
    main()
