#!/usr/bin/env python3
"""
Colab smoke test: GPU check → Samanantar download → Clean → Train sinusoidal
Usage: python scripts/colab_smoke_test.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

def check_gpu():
    """Check if GPU (T4) is available"""
    print("\n" + "="*60)
    print("STEP 1: Checking GPU")
    print("="*60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Device Count: {torch.cuda.device_count()}")
            return True
        else:
            print("❌ GPU NOT available - CPU mode only")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def download_samanantar_small():
    """Download small Samanantar sample"""
    print("\n" + "="*60)
    print("STEP 2: Downloading Samanantar (small subset)")
    print("="*60)
    
    from datasets import load_dataset
    
    output_dir = Path("datasets/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "samanantar_hi_en_small.tsv"
    
    if output_file.exists():
        print(f"✅ File already exists: {output_file}")
        return str(output_file)
    
    try:
        print("Downloading from ai4bharat/samanantar...")
        ds = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=False)
        print(f"✅ Downloaded {len(ds)} pairs")
        
        # Take only first 500 for smoke test
        test_size = min(500, len(ds))
        print(f"Taking first {test_size} pairs for smoke test...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                if i >= test_size:
                    break
                # Handle both 'translation' dict and flat columns
                if isinstance(item.get("translation"), dict):
                    src = item["translation"].get("hi", "")
                    tgt = item["translation"].get("en", "")
                else:
                    src = item.get("hi", "")
                    tgt = item.get("en", "")
                
                if src and tgt:
                    f.write(f"{src}\t{tgt}\n")
        
        print(f"✅ Saved to {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None

def clean_data(input_file):
    """Clean and preprocess data"""
    print("\n" + "="*60)
    print("STEP 3: Cleaning and preprocessing data")
    print("="*60)
    
    output_dir = Path("datasets/processed/smoke_test_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run preprocess script
        cmd = [
            sys.executable,
            "data/preprocess_mt.py",
            input_file,
            str(output_dir),
            "--seed", "42",
            "--max-pairs", "500",
            "--max-length", "128"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Data cleaned")
            print(result.stdout)
            return str(output_dir)
        else:
            print(f"❌ Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error cleaning data: {e}")
        return None

def train_sinusoidal(data_dir):
    """Train sinusoidal baseline on small dataset"""
    print("\n" + "="*60)
    print("STEP 4: Training sinusoidal on smoke test data")
    print("="*60)
    
    train_file = Path(data_dir) / "train.tsv"
    val_file = Path(data_dir) / "val.tsv"
    output_dir = Path("results/colab_smoke_test/sinusoidal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = [
            sys.executable,
            "src/train_mt.py",
            str(train_file),
            str(val_file),
            str(output_dir),
            "--num-steps", "500",  # Short training
            "--batch-size", "8",
            "--learning-rate", "3e-4",
            "--max-seq-length", "128",
            "--d-model", "256",
            "--n-layers", "6",
            "--n-heads", "8",
            "--positional-encoding", "sinusoidal",
            "--log-interval", "50",
            "--eval-interval", "100",
            "--seed", "42"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("(This will take 2-5 minutes...)")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ Training complete")
            # Check outputs
            metrics_file = output_dir / "metrics.jsonl"
            best_ckpt = output_dir / "best.pt"
            
            if metrics_file.exists():
                print(f"✅ Metrics file exists: {metrics_file}")
                with open(metrics_file, "r") as f:
                    lines = f.readlines()
                    print(f"   {len(lines)} metric records logged")
            
            if best_ckpt.exists():
                size_mb = best_ckpt.stat().st_size / (1024*1024)
                print(f"✅ Best checkpoint exists: {best_ckpt} ({size_mb:.1f} MB)")
            
            return str(output_dir)
        else:
            print(f"❌ Training error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error training: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("🔥 COLAB SMOKE TEST PIPELINE")
    print("="*60)
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    print(f"Working directory: {repo_root}")
    
    # Step 1: Check GPU
    gpu_available = check_gpu()
    
    # Step 2: Download Samanantar
    samanantar_file = download_samanantar_small()
    if not samanantar_file:
        print("❌ Failed to download Samanantar")
        return False
    
    # Step 3: Clean data
    data_dir = clean_data(samanantar_file)
    if not data_dir:
        print("❌ Failed to clean data")
        return False
    
    # Step 4: Train sinusoidal
    train_dir = train_sinusoidal(data_dir)
    if not train_dir:
        print("❌ Failed to train model")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("✅ SMOKE TEST COMPLETE")
    print("="*60)
    print(f"GPU Available: {gpu_available}")
    print(f"Samanantar file: {samanantar_file}")
    print(f"Data dir: {data_dir}")
    print(f"Training output: {train_dir}")
    print("\n✅ Pipeline is working fine! Ready for full baselines.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
