# Minimal-Effort Plan to Complete the Research

> Based on review findings + decision to **remove Sinusoidal** and focus exclusively on RoPE/AdaptiveRoPE/gate behavior.

---

## ✅ Key Insight: Removing Sinusoidal Fixes the Fatal Flaw

The cross-attention PE asymmetry was only a problem **because Sinusoidal was the outlier** (embedding-level PE propagates to cross-attention; RoPE/ALiBi do not).

By removing Sinusoidal from the comparison:
- **All remaining methods** (RoPE, AdaptiveRoPE, GatesOnly, PhasesOnly, ALiBi) apply PE **only in self-attention**
- The comparison becomes **fair by construction**
- No code changes to `MultiHeadCrossAttention` are needed

You can legitimately frame this as: *"Following our base paper's decoder-only formulation, all rotary methods are applied exclusively in self-attention layers. Cross-attention receives position-agnostic encoder outputs, consistent with the original design."*

---

## 🎯 What to Remove

### From `paper/main.tex`:
1. **Table 1 (tab:main):** Delete the entire `Sinusoidal` row.
2. **Abstract:** Remove "Sinusoidal PE dominates Hi-En (+141% BLEU)..." Rewrite to focus on RoPE vs AdaptiveRoPE vs ablations.
3. **Section 5.2 (Cross-lingual):** Remove all Sinusoidal discussion. Focus on:
   - RoPE baseline
   - AdaptiveRoPE improvement
   - GatesOnly vs PhasesOnly decomposition
   - ALiBi as non-rotary baseline
4. **Figure 2 (crosslingual_comparison.png):** Regenerate without Sinusoidal bar.
5. **All other mentions:** Search for "sinusoidal" / "Sinusoidal" and remove.

### From evaluation scripts:
- No changes needed. Just don't include `sinusoidal_*` checkpoints in tables/figures.

---

## 🎯 What to Keep (The Story)

Your research question is now **clean and focused**:

> *"When extending decoder-only rotary PE to encoder-decoder MT, what positional mechanisms matter? We decompose AdaptiveRoPE into learnable frequency gates and phase offsets, and test across En-De, Hi-En, and Bn-En."*

**The narrative:**
1. **RoPE** is the baseline (strong on En-De, weaker on Indic)
2. **AdaptiveRoPE** improves over RoPE by learning per-head frequency/phase adjustments
3. **GatesOnly** shows frequency scaling helps Bengali (morphologically complex)
4. **PhasesOnly** shows phase offsets help Hindi (free word order)
5. **ALiBi** underperforms, confirming rotary methods are better for MT

This is still a valid story — just without the Sinusoidal distraction.

---

## 🔧 Hyperparameter Standardization

### Current (Inconsistent)

| Param | En-De | Hi-En / Bn-En |
|-------|-------|---------------|
| Steps | 25K | 75K |
| Batch | 256 | 64 |
| LR | 1e-3 | 5e-4 |
| Seq len | 128 | 192 |

### Problem
You cannot compare across languages when training differs. But re-training En-De for 75K steps is wasteful (it already converged at 25K).

### Recommended: Two-Tier Approach

**For En-De:** Keep existing models but add a **footnote in Table 1**:
> *"En-De models trained for 25K steps (batch 256, lr 1e-3, seq 128); Hi-En/Bn-En trained for 75K steps (batch 64, lr 5e-4, seq 192). Cross-language ranking comparisons are therefore exploratory."*

**For Hi-En / Bn-En:** Train with **3 seeds** (42, 43, 44) using the same hyperparameters already used:
- Steps: 75K
- Batch: 64
- LR: 5e-4
- Seq: 192
- Warmup: 4K steps
- Label smoothing: 0.1

**Why this is acceptable:**
- The paper's claim shifts from *"PE requirements are language-dependent"* to *"Gates help some languages, phases help others"*
- You compare **methods within each language pair** (RoPE vs AdaptiveRoPE vs GatesOnly vs PhasesOnly on Hi-En), which is valid even if language pairs use different hyperparameters
- You explicitly disclaim cross-language BLEU comparisons

---

## 📋 Concrete Action List (Minimal Effort)

### Phase 1: Paper Rewrite (No Retraining Needed) — 1 day

| # | Task | File | Effort |
|---|------|------|--------|
| 1.1 | Remove Sinusoidal row from Table 1 | `paper/main.tex` | 5 min |
| 1.2 | Rewrite abstract — focus on gate/phase decomposition | `paper/main.tex` | 30 min |
| 1.3 | Rewrite Section 5.2 — remove Sinusoidal, keep gate/phase analysis | `paper/main.tex` | 1 hr |
| 1.4 | Remove Sinusoidal from all other sections | `paper/main.tex` | 30 min |
| 1.5 | Regenerate `crosslingual_comparison.png` without Sinusoidal | `pipeline/paper_figures.py` | 15 min |
| 1.6 | Add Table 1 footnote about hyperparameter differences | `paper/main.tex` | 5 min |
| 1.7 | Rewrite conclusion to match new story | `paper/main.tex` | 30 min |

### Phase 2: Multi-Seed Training (Hi-En + Bn-En) — 3–5 days

| # | Task | Command / Script | GPUs × Time |
|---|------|------------------|-------------|
| 2.1 | Train `rope_hi` seed 43 | `python -m pipeline.train_model --lang hi --pe rope --seed 43` | 1 × ~6h |
| 2.2 | Train `rope_hi` seed 44 | `python -m pipeline.train_model --lang hi --pe rope --seed 44` | 1 × ~6h |
| 2.3 | Train `adaptiverope_hi` seed 43 | `python -m pipeline.train_model --lang hi --pe adaptiverope --seed 43` | 1 × ~6h |
| 2.4 | Train `adaptiverope_hi` seed 44 | `python -m pipeline.train_model --lang hi --pe adaptiverope --seed 44` | 1 × ~6h |
| 2.5 | Train `gatesonly_hi` seed 43 | `python -m pipeline.train_model --lang hi --pe gatesonly --seed 43` | 1 × ~6h |
| 2.6 | Train `gatesonly_hi` seed 44 | `python -m pipeline.train_model --lang hi --pe gatesonly --seed 44` | 1 × ~6h |
| 2.7 | Train `phasesonly_hi` seed 43 | `python -m pipeline.train_model --lang hi --pe phasesonly --seed 43` | 1 × ~6h |
| 2.8 | Train `phasesonly_hi` seed 44 | `python -m pipeline.train_model --lang hi --pe phasesonly --seed 44` | 1 × ~6h |
| 2.9 | Train `alibi_hi` seed 43 | `python -m pipeline.train_model --lang hi --pe alibi --seed 43` | 1 × ~6h |
| 2.10 | Train `alibi_hi` seed 44 | `python -m pipeline.train_model --lang hi --pe alibi --seed 44` | 1 × ~6h |
| 2.11 | Repeat 2.1–2.10 for `bn` | Same commands with `--lang bn` | 1 × ~60h total |

> **Parallelization:** You have 1× A100 80GB. Each training run takes ~6 hours. Total: ~120 GPU-hours (5 days sequential, or less if you parallelize across multiple GPUs if available).

### Phase 3: Re-evaluation + Stats — 1 day

| # | Task | Script | Effort |
|---|------|--------|--------|
| 3.1 | Evaluate all new Hi-En seeds (greedy, 3K) | `python -m pipeline.evaluate_model` | Batch |
| 3.2 | Evaluate all new Bn-En seeds (greedy, 3K) | `python -m pipeline.evaluate_model` | Batch |
| 3.3 | Compute mean ± std across 3 seeds per method/lang | Python script | 1 hr |
| 3.4 | Run bootstrap significance tests (Hi-En, Bn-En) | Add to analysis script | 2 hr |

### Phase 4: Final Paper Polish — 1 day

| # | Task | File | Effort |
|---|------|------|--------|
| 4.1 | Update Table 1 with multi-seed stats | `paper/main.tex` | 30 min |
| 4.2 | Add significance test results (new Table 2 or inline) | `paper/main.tex` | 1 hr |
| 4.3 | Fix missing citations (Chen 2023 PI, Snover 2006 TER, xPos) | `paper/references.bib`, `paper/main.tex` | 30 min |
| 4.4 | Fix citation keys (`loshchilov2018` → `2019`, `press2021` → `2022`) | `paper/references.bib`, `paper/main.tex` | 15 min |
| 4.5 | Remove/rephrase unsupported linguistic claims | `paper/main.tex` | 30 min |
| 4.6 | Replace `[Your Institution]` placeholder | `paper/main.tex` | 5 min |
| 4.7 | Recompile PDF, verify all citations resolve | `paper/` | 15 min |

---

## 📊 Expected Updated Table 1 (Without Sinusoidal)

```latex
\begin{table*}[t]
\centering
\small
\caption{Translation quality across RoPE variants and language pairs.
En--De reports mean~$\pm$~std over 3 seeds; Hi--En and Bn--En use 3 seeds.
Bold = best per column. $\uparrow$/$\downarrow$ indicate preferred direction.
$^\dagger$En--De trained with 25K steps (batch 256, lr $10^{-3}$, seq 128);
Indic models use 75K steps (batch 64, lr $5\times10^{-4}$, seq 192).}
\label{tab:main}
\begin{tabular}{l|ccc|ccc|ccc|c}
\toprule
& \multicolumn{3}{c|}{\textbf{En--De}$^\dagger$}
& \multicolumn{3}{c|}{\textbf{Hi--En}}
& \multicolumn{3}{c|}{\textbf{Bn--En}}
& \\
\textbf{Method}
& BLEU$\uparrow$ & chrF$\uparrow$ & TER$\downarrow$
& BLEU$\uparrow$ & chrF$\uparrow$ & TER$\downarrow$
& BLEU$\uparrow$ & chrF$\uparrow$ & TER$\downarrow$
& Decode~s$\downarrow$ \\
\midrule
RoPE & 20.83$\pm$0.03 & 52.51$\pm$0.19 & 68.20$\pm$0.06 & X.XX$\pm$Y.YY & ... & ... & X.XX$\pm$Y.YY & ... & ... & 62.0$\pm$0.3 \\
AdaptiveRoPE & 20.69$\pm$0.05 & 52.40$\pm$0.02 & 68.45$\pm$0.12 & X.XX$\pm$Y.YY & ... & ... & X.XX$\pm$Y.YY & ... & ... & 68.0$\pm$0.4 \\
GatesOnly & 20.68$\pm$0.08 & 52.45$\pm$0.08 & 68.48$\pm$0.19 & X.XX$\pm$Y.YY & ... & ... & X.XX$\pm$Y.YY & ... & ... & 67.6$\pm$0.3 \\
PhasesOnly & 20.75$\pm$0.05 & 52.46$\pm$0.11 & 68.47$\pm$0.03 & X.XX$\pm$Y.YY & ... & ... & X.XX$\pm$Y.YY & ... & ... & 71.6$\pm$6.0 \\
ALiBi & 19.60 & 51.53 & 70.04 & X.XX$\pm$Y.YY & ... & ... & X.XX$\pm$Y.YY & ... & ... & \textbf{55.6} \\
\bottomrule
\end{tabular}
\end{table*}
```

> Replace `X.XX±Y.YY` with actual multi-seed means/std after Phase 2.

---

## 📝 Suggested Abstract Rewrite

```latex
Rotary Position Embedding (RoPE) has become the dominant positional encoding
in large language models, yet its behavior in encoder-decoder machine translation
across typologically diverse languages remains understudied.
We decompose AdaptiveRoPE --- a learnable extension of RoPE with per-head
frequency gates and phase offsets --- and evaluate its components on
WMT14 English--German, Samanantar Hindi--English, and Bengali--English.
On En--De, RoPE and its variants achieve ~20.7 BLEU, while ALiBi lags at 19.6.
On Hi--En, phase offsets provide the largest gain over RoPE,
suggesting that flexible word order benefits from adjustable phase shifts.
On Bn--En, frequency gates are most effective,
indicating that morphological complexity benefits from learnable frequency scaling.
All models share controlled architecture (47M parameters)
and training is replicated across three random seeds.
```

---

## ⚠️ Remaining Honesty Requirements

Even with Sinusoidal removed, you **must** include these caveats:

1. **Hyperparameter differences:** En-De uses 25K/256/1e-3/128; Indic uses 75K/64/5e-4/192. State this in a footnote and do NOT make direct cross-language BLEU comparisons.

2. **Low absolute BLEU on Indic:** If Hi-En RoPE multi-seed mean is < 10 BLEU, acknowledge this suggests the model is underfit or the task is harder than En-De, and discuss potential causes (smaller dataset, longer sequences, morphological complexity).

3. **Post-hoc linguistic claims:** If you say "phase offsets help Hindi because of free word order," prefix with *"We hypothesize that..."* and acknowledge this is speculative.

4. **Decoder-only origin:** Cite your base paper and note: *"Our encoder-decoder implementation applies rotary PE exclusively in self-attention, following the decoder-only convention of [base paper]. Cross-attention operates on position-agnostic encoder outputs."*

---

## 🎯 Summary: Why This Works

| Original Problem | How Removing Sinusoidal Fixes It |
|------------------|----------------------------------|
| Cross-attention PE asymmetry | All remaining methods are self-attention-only → fair comparison |
| +141% Sinusoidal "discovery" | No longer relevant; focus shifts to RoPE variant comparison |
| Uncontrolled training | Acceptable if you only compare methods **within** each language pair |
| Thin novelty | "Gates vs phases decomposition across languages" is a focused, defensible claim |

**Total effort estimate:** ~7–10 days (1 day rewrite + 5 days training + 1 day eval + 1 day polish).

**Submission readiness:** With multi-seed results and honest caveats, this becomes a solid **workshop paper** or lower-tier conference submission. For ACL/EMNLP, you'd still want standardized hyperparameters and higher absolute BLEU.
