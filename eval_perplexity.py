import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from model import GPT


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(key)
        if token:
            return token
    return None


def parse_context_lengths(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths:
        raise ValueError("At least one context length is required.")
    if any(length <= 1 for length in lengths):
        raise ValueError("All context lengths must be > 1.")
    return sorted(set(lengths))


def get_tokenizer(cache_dir: str) -> GPT2TokenizerFast:
    token = _hf_token()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return GPT2TokenizerFast.from_pretrained("gpt2", token=token, cache_dir=str(cache_path))


def _concat_text(examples: list[str]) -> str:
    cleaned = [line for line in examples if line and not line.isspace()]
    return "\n\n".join(cleaned)


def get_or_create_validation_tokens(data_cache_dir: str, tokenizer: GPT2TokenizerFast) -> torch.Tensor:
    cache_path = Path(data_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    val_path = cache_path / "wikitext2_val_gpt2_ids.pt"
    if val_path.exists():
        return torch.load(val_path, map_location="cpu").long()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    val_text = _concat_text(dataset["validation"]["text"])
    val_ids = torch.tensor(
        tokenizer(val_text, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )
    torch.save(val_ids, val_path)
    return val_ids


@torch.no_grad()
def perplexity_sliding_window(
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
    parser.add_argument("--tokenizer_cache", type=str, default=".cache/hf_tokenizer")
    parser.add_argument("--data_cache", type=str, default=".cache/wikitext2_gpt2")
    parser.add_argument("--context_lengths", type=str, default="512,1024,2048")
    parser.add_argument("--max_eval_tokens", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    context_lengths = parse_context_lengths(args.context_lengths)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    tokenizer = get_tokenizer(args.tokenizer_cache)
    data = get_or_create_validation_tokens(args.data_cache, tokenizer)
    if args.max_eval_tokens > 0 and data.size(0) > args.max_eval_tokens:
        data = data[: args.max_eval_tokens]

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    config = checkpoint.get("config", {})
    vocab_size = int(config.get("vocab_size", tokenizer.vocab_size))

    if int(data.max().item()) >= vocab_size:
        raise ValueError(
            f"Found token id >= vocab_size ({vocab_size}). "
            "Checkpoint must be trained with GPT-2 tokenizer (vocab_size=50257)."
        )

    model = GPT(
        vocab_size=vocab_size,
        d_model=int(config.get("d_model", 256)),
        n_layers=int(config.get("n_layers", 4)),
        n_heads=int(config.get("n_heads", 8)),
        max_seq_len=max(int(config.get("max_seq_len", 512)), max(context_lengths)),
        use_as_rope=bool(config.get("use_as_rope", False)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for context_len in context_lengths:
        ppl = perplexity_sliding_window(
            model=model,
            data=data,
            context_len=context_len,
            device=args.device,
        )
        print(f"context={context_len} ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
