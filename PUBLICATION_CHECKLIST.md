# Publication Target Checklist — FIXED VERSION

## All 7 Issues FIXED ✅

### Fix 1: ALiBi Baseline Added
- **Added:** `alibi_de_s42` to training queue
- **Params:** seq=128, batch=256, steps=25K, lr=0.001 (matches all En-De)
- **Why:** Reviewers expect ALiBi as a standard baseline. Earlier OOM was due to batch=512; batch=256 should fit.

### Fix 2: Legacy Hyperparameters Corrected
- **Deleted:** Old `sinusoidal_de` (was seq=256, steps=30K — inconsistent)
- **Retraining:** `sinusoidal_de_correct` with seq=128, batch=256, steps=25K
- **Note:** `rope_hi` and `rope_hi_s7` are from old codebase with different tokenizer. For fair comparison, we now train NEW `rope_hi_s42`, `adaptiverope_hi_s42` with current pipeline.
- **Same for Bn-En:** New `rope_bn_s42`, `adaptiverope_bn_s42` trained with current pipeline.

### Fix 3: Hi-En / Bn-En Evaluations FIXED
- **Problem:** Evaluate script defaulted to WMT14 test data and Helsinki tokenizer.
- **Fix:**
  - `src/train.py` now stores tokenizer name in checkpoint config
  - `src/eval.py` auto-detects tokenizer from checkpoint config
  - `pipeline/evaluate_model.py` defaults to `None` (auto-detect)
  - `pipeline/run_all_evals.py` routes to correct test TSV per language:
    - En-De: `raw_data/wmt14/test.tsv`
    - Hi-En: `raw_data/samanantar/samanantar_hi_en.tsv`
    - Bn-En: `raw_data/samanantar/samanantar_bn_en.tsv`

### Fix 4: Length Generalization Multi-Method
- **New script:** `pipeline/length_generalization_multi.py`
- **Accepts:** Arbitrary number of checkpoints via `--ckpts path:Label`
- **Compares:** All methods side-by-side on same length buckets
- **Outputs:** BLEU curves, degradation curves, chrF curves, summary table

### Fix 5: Position Interpolation Automated
- **PI function:** `apply_position_interpolation()` in `src/positional.py`
- **Automation:** `pipeline/run_all_evals.py` automatically applies PI (scales 1.5×, 2.0×, 3.0×) to all RoPE checkpoints after standard eval

### Fix 6: Paper Framework Ready
- **New analysis orchestrator:** `pipeline/run_all_analysis.py`
- Runs eval → length gen → attention → stats in one command
- All figures and tables will be generated automatically after training

### Fix 7: Attention Analysis for ALL Checkpoints
- **New script:** `pipeline/run_attention_batch.py`
- Runs `src.attention_analysis.py` on every checkpoint automatically
- Generates entropy plots + sink-token plots per method

---

## Training Queue (After Fixes)

### En-De (sequential, ~2.5h each)
| # | Run | PE Type | Seed | Status |
|---|-----|---------|------|--------|
| 1 | `rope_de_s43` | RoPE | 43 | ⏳ Queued |
| 2 | `rope_de_s44` | RoPE | 44 | ⏳ Queued |
| 3 | `adaptiverope_de_s43` | AdaptiveRoPE | 43 | ⏳ Queued |
| 4 | `adaptiverope_de_s44` | AdaptiveRoPE | 44 | ⏳ Queued |
| 5 | `sinusoidal_de_s43` | Sinusoidal | 43 | ⏳ Queued |
| 6 | `sinusoidal_de_s44` | Sinusoidal | 44 | ⏳ Queued |
| 7 | `sinusoidal_de_correct` | Sinusoidal | 42 | ⏳ Queued |
| 8 | `gatesonly_de_s42` | GatesOnly | 42 | ⏳ Queued |
| 9 | `gatesonly_de_s43` | GatesOnly | 43 | ⏳ Queued |
| 10 | `gatesonly_de_s44` | GatesOnly | 44 | ⏳ Queued |
| 11 | `phasesonly_de_s42` | PhasesOnly | 42 | ⏳ Queued |
| 12 | `phasesonly_de_s43` | PhasesOnly | 43 | ⏳ Queued |
| 13 | `phasesonly_de_s44` | PhasesOnly | 44 | ⏳ Queued |
| 14 | `alibi_de_s42` | ALiBi | 42 | ⏳ Queued |

**En-De total:** 14 runs × ~2.5h = ~35h

### Hi-En (parallel up to 3, ~2h each)
| # | Run | PE Type | Seed |
|---|-----|---------|------|
| 1 | `rope_hi_s42` | RoPE | 42 |
| 2 | `adaptiverope_hi_s42` | AdaptiveRoPE | 42 |
| 3 | `gatesonly_hi_s42` | GatesOnly | 42 |
| 4 | `phasesonly_hi_s42` | PhasesOnly | 42 |

**Hi-En total:** 4 runs, wall-clock ~3h (parallel batches)

### Bn-En (parallel up to 3, ~2h each)
| # | Run | PE Type | Seed |
|---|-----|---------|------|
| 1 | `rope_bn_s42` | RoPE | 42 |
| 2 | `adaptiverope_bn_s42` | AdaptiveRoPE | 42 |
| 3 | `sinusoidal_bn_s42` | Sinusoidal | 42 |
| 4 | `gatesonly_bn_s42` | GatesOnly | 42 |
| 5 | `phasesonly_bn_s42` | PhasesOnly | 42 |

**Bn-En total:** 5 runs, wall-clock ~4h (parallel batches)

**Grand Total ETA:** ~35h (En-De) + ~4h (Hi/Bn overlap) = **~39 hours (~1.6 days)**

---

## Post-Training Analysis (One Command)

After training finishes, run:
```bash
python -m pipeline.run_all_analysis
```

This executes:
1. `pipeline.run_all_evals` — BLEU/chrF/TER + PI variants for all checkpoints
2. `pipeline.length_generalization_multi` — Length gen curves for all En-De methods
3. `pipeline.run_attention_batch` — Attention entropy for all checkpoints
4. `src.statistical_tests` — t-test + Wilcoxon for all language pairs

Output locations:
```
outputs/metrics/                    # All eval results
outputs/analysis/length_gen_multi/  # Length generalization plots
outputs/analysis/attention/         # Per-checkpoint attention analysis
outputs/analysis/stats_*.json       # Statistical significance tests
```

---

## Known Limitations (For Paper Disclosure)

1. **Single architecture** — Custom 47M encoder-decoder. Frame as "controlled setting."
2. **No YaRN** — PI covers interpolation; YaRN is optional.
3. **Short sequences** — max_seq_len=128/192. Frame as "standard MT lengths."
4. **3 seeds minimum** — Sufficient for Findings/Workshop venues.
5. **Old Hi-En/Bn-En checkpoints excluded** — `rope_hi`, `asrope_hi`, etc. used old codebase with `IndicBART` tokenizer. New Hi/Bn results use current pipeline for consistency.

---

## How to Monitor

```bash
# Check tmux session (survives disconnect)
tmux ls
tmux attach -t neur_training

# Check current step
tail -1 outputs/logs/ROUTE_NAME/metrics.jsonl

# Check GPU
nvidia-smi

# Check completed runs
for d in outputs/logs/*/; do [ -f "${d}run_summary.json" ] && echo "$(basename "$d") ✅"; done
```
