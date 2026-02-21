import argparse
import math
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

from model import GPT

GPT2_VOCAB_URL = "https://huggingface.co/gpt2/resolve/main/vocab.json"
GPT2_MERGES_URL = "https://huggingface.co/gpt2/resolve/main/merges.txt"
TINY_SHAKESPEARE_URL = "https://huggingface.co/datasets/karpathy/tiny_shakespeare/resolve/main/tiny_shakespeare.txt"


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


def load_tiny_shakespeare_tokens(tokenizer: ByteLevelBPETokenizer) -> torch.Tensor:
    ds = load_dataset("text", data_files=TINY_SHAKESPEARE_URL, split="train")
    text = "\n".join(ds["text"])
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    tokenizer = build_gpt2_tokenizer(args.tokenizer_cache)
    data = load_tiny_shakespeare_tokens(tokenizer)

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
        max_seq_len=max(int(config.get("max_seq_len", 1024)), 2048),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for context_len in (512, 1024, 2048):
        ppl = perplexity_sliding_no_double_count(
            model=model,
            data=data,
            context_len=context_len,
            device=args.device,
        )
        print(f"context={context_len} ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
