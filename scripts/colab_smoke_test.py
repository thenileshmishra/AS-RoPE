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
    """Download small Samanantar sample - using hardcoded minimal data for faster testing"""
    print("\n" + "="*60)
    print("STEP 2: Creating minimal test dataset")
    print("="*60)
    
    output_dir = Path("datasets/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "samanantar_hi_en_small.tsv"
    
    if output_file.exists() and output_file.stat().st_size > 100:
        print(f"✅ File already exists: {output_file}")
        return str(output_file)
    
    try:
        # Minimal dataset for fast smoke testing
        pairs = [
            ("नमस्ते, आप कैसे हैं?", "Hello, how are you?"),
            ("मेरा नाम राज है।", "My name is Raj."),
            ("क्या आप अंग्रेजी बोलते हैं?", "Do you speak English?"),
            ("भारत एक सुंदर देश है।", "India is a beautiful country."),
            ("मुझे आपसे मिलकर खुशी हुई।", "It was nice meeting you."),
            ("यह पुस्तक बहुत अच्छी है।", "This book is very good."),
            ("मैं कल आपसे फिर मिलूंगा।", "I will meet you again tomorrow."),
            ("आपका स्वागत है।", "You are welcome."),
            ("मेरा परिवार बहुत बड़ा है।", "My family is very big."),
            ("क्या आप मुझे मदद कर सकते हैं?", "Can you help me?"),
            ("बाजार से सब्जियां लानी हैं।", "I need to buy vegetables from the market."),
            ("आज का मौसम बहुत गर्म है।", "Today's weather is very hot."),
            ("मैं पढ़ाई में बहुत अच्छा हूं।", "I am very good at studies."),
            ("यह फिल्म देखना चाहिए।", "You should watch this movie."),
            ("मेरे पास कोई समय नहीं है।", "I don't have any time."),
            ("रात को सोना स्वास्थ्य के लिए अच्छा है।", "Sleeping at night is good for health."),
            ("मुझे खेल खेलना बहुत पसंद है।", "I love playing sports."),
            ("दिल्ली भारत की राजधानी है।", "Delhi is the capital of India."),
            ("क्या मैं आपका फोटो ले सकता हूँ?", "Can I take your photo?"),
            ("कल हम पार्क में जाएंगे।", "Tomorrow we will go to the park."),
        ]
        
        with open(output_file, "w", encoding="utf-8") as f:
            for src, tgt in pairs:
                f.write(f"{src}\t{tgt}\n")
        
        print(f"✅ Created test dataset with {len(pairs)} pairs")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_data(input_file):
    """Clean and preprocess data"""
    print("\n" + "="*60)
    print("STEP 3: Cleaning and preprocessing data")
    print("="*60)
    
    output_dir = Path("datasets/processed/smoke_test_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run preprocess script with PYTHONPATH set
        env = os.environ.copy()
        repo_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(repo_root)
        
        cmd = [
            sys.executable,
            "data/preprocess_mt.py",
            input_file,
            str(output_dir),
            "--seed", "42",
            "--max-length", "128"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        
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
        # Set PYTHONPATH for subprocess
        env = os.environ.copy()
        repo_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(repo_root)
        
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        
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
