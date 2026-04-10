"""
Colab Training Pipeline: GPU Check → Data Loading → Preprocessing → Training → Evaluation
Optimized for running inside Colab notebooks with minimal setup.
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# SECTION 1: GPU VERIFICATION
# ============================================================================

def verify_gpu():
    """Check if T4 GPU is available in Colab."""
    print("=" * 70)
    print("STEP 1: GPU VERIFICATION")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"✓ GPU Available: {device_name}")
        print(f"✓ GPU Count: {device_count}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        
        # Check memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Total GPU Memory: {total_mem:.2f} GB")
        
        return torch.device("cuda")
    else:
        print("✗ NO GPU DETECTED! Please ensure T4 GPU is enabled in Colab.")
        print("  Go to: Runtime → Change runtime type → GPU (T4)")
        raise RuntimeError("GPU not available")


# ============================================================================
# SECTION 2: ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Setup Python path and required packages."""
    print("\n" + "=" * 70)
    print("STEP 2: ENVIRONMENT SETUP")
    print("=" * 70)
    
    # Add repo to path if needed
    repo_path = Path("/content/AS-RoPE")
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))
        print(f"✓ Added repo to path: {repo_path}")
    else:
        print("⚠ AS-RoPE repo not found at /content/AS-RoPE")
        print("  Make sure clone_repo() was called first")
    
    # Verify packages
    required_packages = ["torch", "datasets", "transformers", "tokenizers"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} available")
        except ImportError:
            print(f"✗ {pkg} NOT available - installing...")
            os.system(f"pip install -q {pkg}")


# ============================================================================
# SECTION 3: DATA LOADING
# ============================================================================

def download_and_clean_samanantar(sample_size: int = 2000, output_dir: str = "/content/datasets"):
    """Download Samanantar dataset and clean it."""
    print("\n" + "=" * 70)
    print("STEP 3: DOWNLOAD & CLEAN SAMANANTAR")
    print("=" * 70)
    
    from datasets import load_dataset
    
    raw_out = Path(output_dir) / "raw" / "samanantar_hi_en_raw.tsv"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Samanantar dataset (sample_size={sample_size})...")
    data = load_dataset("ai4bharat/samanantar", "hi", split="train")
    data = data.shuffle(seed=42).select(range(sample_size))
    
    # Detect format
    src_col, tgt_col = None, None
    for ex in data.select(range(min(100, len(data)))):
        if "translation" in ex:
            tr = ex["translation"]
            if isinstance(tr, dict) and "hi" in tr and "en" in tr:
                src_col, tgt_col = "translation", "translation"
                break
    
    if src_col is None:
        raise RuntimeError("Could not detect translation columns in Samanantar")
    
    print(f"✓ Detected format: {src_col} → {tgt_col}")
    
    # Clean and deduplicate
    seen = set()
    kept = 0
    
    with open(raw_out, "w", encoding="utf-8") as f:
        for ex in data:
            try:
                if src_col == "translation":
                    tr = ex["translation"]
                    src = tr.get("hi", "").strip()
                    tgt = tr.get("en", "").strip()
                else:
                    src = ex.get(src_col, "").strip()
                    tgt = ex.get(tgt_col, "").strip()
                
                # Cleaning rules
                if not src or not tgt:
                    continue
                if len(src.split()) < 2 or len(tgt.split()) < 2:
                    continue
                if len(src.split()) > 256 or len(tgt.split()) > 256:
                    continue
                
                key = (src, tgt)
                if key in seen:
                    continue
                seen.add(key)
                
                f.write(f"{src}\t{tgt}\n")
                kept += 1
            except Exception:
                continue
    
    print(f"✓ Cleaned and deduplicated: {kept} pairs → {raw_out}")
    return str(raw_out)


# ============================================================================
# SECTION 4: PREPROCESSING
# ============================================================================

def preprocess_dataset(raw_tsv: str, output_dir: str = "/content/datasets/processed/hien_v1", 
                      max_len: int = 128, tokenizer_name: str = "gpt2"):
    """Preprocess raw TSV into train/val/test splits with tokenization."""
    print("\n" + "=" * 70)
    print("STEP 4: PREPROCESS DATASET")
    print("=" * 70)
    
    from transformers import AutoTokenizer
    from data.mt_dataset import deterministic_split
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw pairs
    print(f"Loading raw TSV: {raw_tsv}")
    pairs = []
    with open(raw_tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, tgt = line.split("\t", maxsplit=1)
            if src and tgt:
                pairs.append((src, tgt))
    
    print(f"✓ Loaded {len(pairs)} pairs")
    
    # Split deterministically
    train_pairs, val_pairs, test_pairs = deterministic_split(
        pairs, 
        train_ratio=0.9, 
        val_ratio=0.05, 
        seed=42
    )
    
    print(f"✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Write TSVs
    for name, pairs_subset in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        tsv_path = output_dir / f"{name}.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            for src, tgt in pairs_subset:
                f.write(f"{src}\t{tgt}\n")
        print(f"✓ Wrote {tsv_path}")
    
    # Tokenize and cache
    print("Tokenizing and caching...")
    for name, pairs_subset in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        src_ids_list = []
        tgt_ids_list = []
        for src, tgt in pairs_subset:
            s = tokenizer(src, max_length=max_len, padding="max_length", 
                          truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
            t = tokenizer(tgt, max_length=max_len, padding="max_length", 
                          truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
            src_ids_list.append(s)
            tgt_ids_list.append(t)
        
        if src_ids_list:
            src_ids = torch.stack(src_ids_list)
            tgt_ids = torch.stack(tgt_ids_list)
        else:
            src_ids = torch.zeros(0, max_len, dtype=torch.long)
            tgt_ids = torch.zeros(0, max_len, dtype=torch.long)
        
        pt_path = output_dir / f"{name}.pt"
        torch.save({"src_ids": src_ids, "tgt_ids": tgt_ids}, pt_path)
        print(f"✓ Cached {pt_path} ({src_ids.shape[0]} examples)")
    
    # Save metadata
    metadata = {
        "total_loaded": len(pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "test_count": len(test_pairs),
        "max_length": max_len,
        "tokenizer": tokenizer_name,
        "seed": 42
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {meta_path}")
    
    return str(output_dir)


# ============================================================================
# SECTION 5: TRAINING
# ============================================================================

def train_sinusoidal_model(
    train_dir: str = "/content/datasets/processed/hien_v1",
    output_dir: str = "/content/results/sinusoidal_run",
    num_steps: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    max_len: int = 128,
    device = None
):
    """Train sinusoidal positional encoding baseline."""
    print("\n" + "=" * 70)
    print("STEP 5: TRAIN SINUSOIDAL MODEL")
    print("=" * 70)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from src.model import GPT
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = Path(train_dir)
    train_cache = train_dir / "train.pt"
    val_cache = train_dir / "val.pt"
    
    print(f"Loading train/val caches from {train_dir}")
    train_data = torch.load(train_cache)
    val_data = torch.load(val_cache)
    
    train_src_ids = train_data["src_ids"]
    train_tgt_ids = train_data["tgt_ids"]
    val_src_ids = val_data["src_ids"]
    val_tgt_ids = val_data["tgt_ids"]
    
    print(f"✓ Train: {train_src_ids.shape[0]} examples")
    print(f"✓ Val: {val_src_ids.shape[0]} examples")
    
    # Create model
    print("Creating GPT model with sinusoidal encoding...")
    model = GPT(
        vocab_size=50257,
        max_seq_len=max_len,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        positional_encoding="sinusoidal",
        dropout=0.1
    ).to(device)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Build MT examples (source + target with target-only masking)
    def build_mt_example(src_ids, tgt_ids, tokenizer_eos=50256):
        """Concatenate src and tgt, build labels with target-only masking."""
        src_len = (src_ids != 0).sum().item()
        tgt_len = (tgt_ids != 0).sum().item()
        
        # Concatenate
        combined = torch.cat([src_ids, tgt_ids])
        
        # Build labels: -100 for source (masked), token_id for target
        labels = torch.full_like(combined, -100)
        labels[src_len:src_len + tgt_len] = tgt_ids[:tgt_len]
        
        return combined, labels
    
    # Dataset wrapper
    class MTDataset(Dataset):
        def __init__(self, src_ids, tgt_ids):
            self.src_ids = src_ids
            self.tgt_ids = tgt_ids
        
        def __len__(self):
            return len(self.src_ids)
        
        def __getitem__(self, idx):
            combined, labels = build_mt_example(self.src_ids[idx], self.tgt_ids[idx])
            return {"input_ids": combined, "labels": labels}
    
    train_dataset = MTDataset(train_src_ids, train_tgt_ids)
    val_dataset = MTDataset(val_src_ids, val_tgt_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def compute_loss(model, batch, device):
        """Compute loss with target-only masking."""
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            input_ids.view(-1),
            reduction="none"
        )
        
        # Mask: only compute loss where labels != -100
        mask = labels.view(-1) != -100
        if mask.sum() > 0:
            loss = loss[mask].mean()
        else:
            loss = loss.mean()
        
        return loss
    
    # Training loop
    print(f"Starting training: {num_steps} steps, batch_size={batch_size}")
    metrics_log = []
    best_val_loss = float("inf")
    
    step = 0
    for epoch in range(10):  # Multiple epochs to hit num_steps
        model.train()
        for batch in train_loader:
            loss = compute_loss(model, batch, device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Validation every 100 steps
            if step % 100 == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_loss = compute_loss(model, val_batch, device)
                        val_losses.append(val_loss.item())
                val_loss_avg = sum(val_losses) / len(val_losses) if val_losses else 0
                
                print(f"Step {step}/{num_steps} | train_loss={loss.item():.4f} | val_loss={val_loss_avg:.4f}")
                
                metrics_log.append({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss_avg
                })
                
                # Save best checkpoint
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    ckpt_path = output_dir / "best.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  ✓ Saved best checkpoint: {ckpt_path}")
                
                model.train()
            
            step += 1
            if step >= num_steps:
                break
        
        if step >= num_steps:
            break
    
    # Save final checkpoint
    last_ckpt = output_dir / "last.pt"
    torch.save(model.state_dict(), last_ckpt)
    print(f"✓ Saved last checkpoint: {last_ckpt}")
    
    # Save metrics log
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for metric in metrics_log:
            f.write(json.dumps(metric) + "\n")
    print(f"✓ Saved metrics: {metrics_path}")
    
    print(f"\n✓ Training Complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Total steps: {step}")
    print(f"  Output dir: {output_dir}")
    
    return str(output_dir)


# ============================================================================
# SECTION 6: EVALUATION
# ============================================================================

def print_summary(train_dir: str, results_dir: str):
    """Print training summary and metrics."""
    print("\n" + "=" * 70)
    print("STEP 6: TRAINING SUMMARY")
    print("=" * 70)
    
    # Load metadata
    metadata_path = Path(train_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print("\nDataset Metadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
    
    # Load metrics
    metrics_path = Path(results_dir) / "metrics.jsonl"
    if metrics_path.exists():
        print("\nTraining Metrics (final 5 steps):")
        metrics = []
        with open(metrics_path) as f:
            for line in f:
                metrics.append(json.loads(line))
        
        for metric in metrics[-5:]:
            print(f"  Step {metric['step']}: train_loss={metric['train_loss']:.4f}, "
                  f"val_loss={metric['val_loss']:.4f}")
    
    # Checkpoint status
    results_path = Path(results_dir)
    if (results_path / "best.pt").exists():
        print(f"\n✓ Best checkpoint exists: {results_path / 'best.pt'}")
    if (results_path / "last.pt").exists():
        print(f"✓ Last checkpoint exists: {results_path / 'last.pt'}")
    
    print("\n✓ PIPELINE COMPLETE!")
    print("=" * 70)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_complete_pipeline(
    sample_size: int = 2000,
    num_steps: int = 1000,
    batch_size: int = 16
):
    """Run the complete Colab pipeline."""
    
    device = verify_gpu()
    setup_environment()
    
    raw_tsv = download_and_clean_samanantar(sample_size=sample_size)
    train_dir = preprocess_dataset(raw_tsv)
    results_dir = train_sinusoidal_model(
        train_dir=train_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        device=device
    )
    
    print_summary(train_dir, results_dir)
    
    return {
        "device": device,
        "raw_tsv": raw_tsv,
        "train_dir": train_dir,
        "results_dir": results_dir,
    }


if __name__ == "__main__":
    run_complete_pipeline(sample_size=2000, num_steps=1000, batch_size=16)
