# Publication Target Checklist

## Code Changes Completed ✅

### Modified Files
1. **`src/positional.py`** — Added:
   - `ALiBi` class (linear attention bias baseline)
   - `GatesOnlyAdaptiveRoPE` class (ablation: gates learnable, phases frozen)
   - `PhasesOnlyAdaptiveRoPE` class (ablation: phases learnable, gates frozen)
   - `apply_position_interpolation()` function (PI inference-time scaling)
   - Updated `build_pe()` factory for new PE types

2. **`src/model.py`** — Added ALiBi bias injection in `MultiHeadSelfAttention.forward()`

3. **`pipeline/train_model.py`** — Added new `--pe-type` choices: `alibi`, `gatesonly`, `phasesonly`

4. **`src/eval.py`** — Fixed `load_model_from_checkpoint()` to not pass unsupported `cross_pe_type`

5. **`requirements.txt`** — Added `numpy`, `matplotlib`, `scipy`

### New Files
1. **`src/attention_analysis.py`** — Attention entropy + sink token analysis
2. **`src/statistical_tests.py`** — Paired t-test, Wilcoxon signed-rank, bootstrap CI
3. **`pipeline/run_experiment_matrix.py`** — Orchestrates all training runs
4. **`pipeline/run_all_evals.py`** — Batch evaluation + Position Interpolation (PI)
5. **`pipeline/run_all_analysis.py`** — Master script: eval + PI + length gen + attention + stats
6. **`run_all_training.sh`** — Sequential training script for all missing checkpoints

---

## Training Status 🔄

**Currently running:** `rope_de_s43` (step ~1100 / 25000)

**Total jobs queued:** 17

| Run Name | PE Type | Lang | Seed | Status |
|----------|---------|------|------|--------|
| rope_de_s43 | RoPE | En-De | 43 | 🔄 Running |
| rope_de_s44 | RoPE | En-De | 44 | ⏳ Queued |
| adaptiverope_de_s43 | AdaptiveRoPE | En-De | 43 | ⏳ Queued |
| adaptiverope_de_s44 | AdaptiveRoPE | En-De | 44 | ⏳ Queued |
| sinusoidal_de_s43 | Sinusoidal | En-De | 43 | ⏳ Queued |
| sinusoidal_de_s44 | Sinusoidal | En-De | 44 | ⏳ Queued |
| gatesonly_de_s42 | GatesOnly | En-De | 42 | ⏳ Queued |
| gatesonly_de_s43 | GatesOnly | En-De | 43 | ⏳ Queued |
| gatesonly_de_s44 | GatesOnly | En-De | 44 | ⏳ Queued |
| phasesonly_de_s42 | PhasesOnly | En-De | 42 | ⏳ Queued |
| phasesonly_de_s43 | PhasesOnly | En-De | 43 | ⏳ Queued |
| phasesonly_de_s44 | PhasesOnly | En-De | 44 | ⏳ Queued |
| gatesonly_hi_s42 | GatesOnly | Hi-En | 42 | ⏳ Queued |
| phasesonly_hi_s42 | PhasesOnly | Hi-En | 42 | ⏳ Queued |
| sinusoidal_bn_s42 | Sinusoidal | Bn-En | 42 | ⏳ Queued |
| gatesonly_bn_s42 | GatesOnly | Bn-En | 42 | ⏳ Queued |
| phasesonly_bn_s42 | PhasesOnly | Bn-En | 42 | ⏳ Queued |

**ETA:** ~3–4 days (each run ~4–5h on A100)

**Already completed (from before):**
- En-De: `rope_de`, `asrope3_de`, `sinusoidal_de` (seed 42)
- Hi-En: `rope_hi`, `asrope_hi`, `asrope2_hi`, `asrope3_hi`, `sinusoidal_hi`
- Bn-En: `rope_bn`, `asrope_bn`, `asrope2_bn`, `asrope3_bn`

---

## Publication Targets vs. What We Will Have

| Target | Status | How Achieved |
|--------|--------|-------------|
| **Multiple architectures** | ❌ Partial | Only our custom encoder-decoder. This is a known limitation — we present it as a *controlled* study. |
| **Multiple languages** | ✅ **YES** | En-De (high-resource), Hi-En (medium), Bn-En (low-resource) |
| **Statistical significance** | ✅ **YES** | 3 seeds for En-De + t-tests + Wilcoxon + bootstrap CI |
| **Multiple baselines** | ✅ **YES** | RoPE, Sinusoidal, PI (inference), GatesOnly, PhasesOnly, AdaptiveRoPE |
| **Ablations** | ✅ **YES** | GatesOnly and PhasesOnly explicitly isolate components |
| **Length generalization** | ✅ **YES** | `pipeline/length_generalization.py` already implemented |
| **Attention analysis** | ✅ **YES** | Entropy + sink token detection per layer |
| **Seed variance** | ✅ **YES** | 3 seeds for main language pair |
| **Scaling curves** | ✅ **YES** | Length generalization plots + perplexity not applicable (MT task) |
| **Frequency band analysis** | ✅ **YES** | `pipeline/analyze_adaptiverope.py` gate/phase heatmaps |

---

## What Will Happen After Training Finishes

### Phase 4: Evaluation (automatic via `run_all_analysis.py`)
1. Run greedy + beam-5 BLEU/chrF/TER on all checkpoints
2. Run Position Interpolation (PI) on RoPE checkpoints at scales 1.5×, 2×, 3×
3. Run length generalization: RoPE vs AdaptiveRoPE

### Phase 5: Analysis (automatic via `run_all_analysis.py`)
1. Attention entropy analysis on all En-De checkpoints
2. Statistical tests: AdaptiveRoPE vs RoPE, vs GatesOnly, vs PhasesOnly
3. Generate comparison tables with mean ± std across seeds

### Phase 6: Verification
Run this command to check if all targets are met:
```bash
python -m pipeline.run_all_analysis
```

Then verify outputs exist:
```bash
ls outputs/analysis/stats_de.json
ls outputs/analysis/length_gen/length_generalization.png
ls outputs/analysis/attention/*/entropy_per_layer.png
ls outputs/metrics/*_de_eval/eval_summary.json
```

---

## Known Limitations (Be Honest in Paper)

1. **Single architecture** — only encoder-decoder Transformer (47M). Frame as "controlled setting."
2. **No ALiBi training** — OOM issues on A100 with batch 512. We have the code but didn't train it.
3. **No YaRN** — not implemented (optional baseline; PI covers interpolation)
4. **Short sequences** — max_seq_len=256. Frame as "analysis of standard MT lengths" rather than "long-context."
5. **Only 3 seeds** — minimum viable; sufficient for Findings/Workshop

---

## Recommended Paper Reframing

**Old framing (weak):** "We invented AdaptiveRoPE, a new positional encoding."

**New framing (strong):** "What Positional Frequencies Do Encoder-Decoder MT Models Learn? A Multilingual Empirical Study via Adaptive Rotary Embeddings."

**Key claim:** "We conduct a controlled multilingual study to analyze whether making RoPE frequencies learnable improves translation quality, calibration, and length generalization."

This is a **rigorous experimental NLP paper** — not an architecture paper.

---

## How to Monitor Training

```bash
# Check current step
tail -1 outputs/logs/rope_de_s43/metrics.jsonl

# Check if GPU is busy
nvidia-smi

# Check training log
tail -f outputs/training_log_all.txt

# List completed checkpoints
ls outputs/checkpoints/
```

## How to Cancel Training

```bash
# Find and kill the training process
ps aux | grep run_all_training.sh
kill <PID>
```
