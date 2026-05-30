"""Evaluate the 3 newly trained missing models and update paper table."""
import json, os, subprocess
from collections import defaultdict

METRICS_DIR = "outputs/metrics"
CHECKPOINTS = {
    "sinusoidal_hi": ("processed_data/test_3k.tsv", "hi"),
    "alibi_hi": ("processed_data/test_3k.tsv", "hi"),
    "alibi_bn": ("processed_data_bn/test_3k.tsv", "bn"),
}

def eval_checkpoint(name, test_tsv, lang):
    ckpt = f"outputs/checkpoints/{name}/best.pt"
    if not os.path.exists(ckpt):
        print(f"SKIP: {name} checkpoint not found")
        return None
    
    out_dir = f"{METRICS_DIR}/{name}_eval"
    if os.path.exists(f"{out_dir}/eval_summary.json"):
        print(f"SKIP: {name} already evaluated")
        with open(f"{out_dir}/eval_summary.json") as f:
            return json.load(f)
    
    print(f"Evaluating {name} ...")
    tokenizer = "ai4bharat/IndicBART" if lang in ("hi", "bn") else None
    cmd = [
        "python3", "-m", "pipeline.evaluate_model",
        "--checkpoint", ckpt,
        "--run-name", f"{name}_eval",
        "--eval-tsv", test_tsv,
        "--beam-size", "1",
        "--batch-size", "32",
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR evaluating {name}:")
        print(result.stderr[-500:])
        return None
    
    with open(f"{out_dir}/eval_summary.json") as f:
        return json.load(f)

if __name__ == "__main__":
    results = {}
    for name, (test_tsv, lang) in CHECKPOINTS.items():
        data = eval_checkpoint(name, test_tsv, lang)
        if data:
            o = data.get("overall", {})
            results[name] = {
                "bleu": o.get("bleu", 0),
                "chrf": o.get("chrf", 0),
                "ter": o.get("ter", 0),
            }
            print(f"  {name}: BLEU={o.get('bleu',0):.2f} chrF={o.get('chrf',0):.2f} TER={o.get('ter',0):.2f}")
    
    if results:
        with open("outputs/missing_model_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved to outputs/missing_model_results.json")
