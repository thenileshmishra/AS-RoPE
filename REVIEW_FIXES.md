# Paper Revision Checklist

> Generated from critical review of `paper/main.tex`, `src/model.py`, `src/positional.py`, and experimental results.
> **Status:** Not submission-ready. Central claims depend on a methodologically flawed comparison.

---

## 🔴 CRITICAL — Must Fix Before Any Submission

### 1. Cross-Attention Positional Encoding Asymmetry
**Problem:**  
Sinusoidal PE is applied at the **embedding level** (`model.py:177, 212-213, 225-226`), so position information propagates into **cross-attention** K/V vectors. RoPE, AdaptiveRoPE, ALiBi, GatesOnly, and PhasesOnly are applied **only in self-attention** (`model.py:35, 138`). `MultiHeadCrossAttention` (`model.py:74-103`) has **no PE at all**.

**Why it matters:**  
This makes the comparison **fundamentally unfair**. Sinusoidal has position-aware cross-attention; all other methods have position-blind cross-attention. The +141% BLEU advantage for Sinusoidal on Hi-En (18.80 vs RoPE 7.78) is likely driven by this asymmetry, not by Hindi's linguistic properties. The paper's central claim — that PE requirements are language-dependent — may collapse entirely.

**Fix (choose ONE):**

| Option | Description | Effort |
|--------|-------------|--------|
| **A. Add RoPE to cross-attention** | Apply RoPE/AdaptiveRoPE rotation to encoder outputs (K) in `MultiHeadCrossAttention`. Decoder query (Q) already has position info from decoder self-attention. | Medium |
| **B. Apply Sinusoidal in attention layers** | Instead of adding Sinusoidal to embeddings, create a `SinusoidalSelfAttention` module that adds PE to Q/K inside attention (like RoPE does). Remove `emb_pe` from `EncoderDecoder`. | Medium |
| **C. Add absolute PE to cross-attention K/V** | For non-Sinusoidal methods, add a learned or sinusoidal position encoding to encoder outputs before they are used as K/V in cross-attention. | Low-Medium |

**Recommended:** Option A — it is the most theoretically sound. Cross-attention should apply RoPE to encoder K (and optionally V) so source positions are available to the decoder.

**Files:** `src/model.py`, `src/positional.py`

**After fixing:** Re-train ALL models (En-De, Hi-En, Bn-En) and re-evaluate. If Sinusoidal still wins on Hi-En, the finding is genuinely novel. If not, the central claim must be rewritten.

---

## 🟡 MAJOR — Significantly Weakens Claims

### 2. Uncontrolled Training Hyperparameters Across Languages
**Problem:**  
Hyperparameters differ drastically across language pairs, yet the paper makes direct cross-lingual comparisons and attributes differences to linguistic properties.

| Parameter | En-De | Hi-En / Bn-En |
|-----------|-------|---------------|
| Steps | 25K | 75K |
| Batch size | 256 | 64 |
| Learning rate | 1e-3 | 5e-4 |
| Max seq length | 128 | 192 |
| Tokenizer | Helsinki-NLP (58K) | IndicBART (64K) |

**Fix:**  
- **Option A (Best):** Standardize ALL hyperparameters across language pairs. Pick one setting (e.g., 75K steps, batch 64, lr 5e-4, seq 192) and re-train En-De models. This is expensive but scientifically valid.
- **Option B (Acceptable):** Keep current results but **remove all cross-lingual causal claims**. Frame the paper as "exploratory" and explicitly state: *"Different hyperparameters preclude direct cross-lingual comparison; we report per-language findings only."*

**Files:** `paper/main.tex` (abstract, introduction, discussion, conclusion), `src/train.py`

---

### 3. Single-Seed Training for Indic Languages
**Problem:**  
En-De uses 3 seeds (42, 43, 44). Hi-En and Bn-En use **1 seed each** (42). No variance estimate. The Table 1 caption says "2–3 seeds" — the "2" is suspicious and unexplained.

**Why it matters:**  
The strong conclusions about Hi-En and Bn-En (e.g., "phase offsets are critical for Hindi") have **no statistical backing**. A single outlier seed could completely flip the ranking.

**Fix:**  
Train all Hi-En and Bn-En models with **at least 2 additional seeds** (e.g., 43, 44). Report mean ± std in Table 1.

**Files:** `pipeline/train_model.py`, `paper/main.tex` (Table 1)

---

### 4. Extremely Low Absolute BLEU Scores
**Problem:**  
- Hi-En RoPE: **7.78 BLEU** — catastrophically low
- Hi-En ALiBi: **6.23 BLEU** — unviable
- En-De best: **20.83 BLEU** — SoTA is >30; basic Transformers reach ~25+

**Why it matters:**  
When baselines are broken, relative improvements are meaningless. "+141% BLEU" sounds dramatic but goes from 7.78 (unusable) to 18.80 (marginal). The low scores suggest training instability, underfitting, or the cross-attention asymmetry causing catastrophic failure for RoPE/ALiBi on long Indic sequences.

**Fix:**  
- First fix the cross-attention asymmetry (Issue #1). Re-evaluate.
- If scores remain low, investigate: learning rate schedule, weight initialization, label smoothing, gradient clipping, or data quality issues.
- Consider increasing model size (currently 47M parameters, d_model=384). This is very small for MT.
- Report **absolute BLEU alongside relative %** so readers understand the baseline is broken.

**Files:** `src/model.py`, `src/train.py`, `paper/main.tex`

---

## 🟠 MODERATE — Weakens Credibility

### 5. Missing Citations

| Missing Citation | Where Needed | Fix |
|------------------|--------------|-----|
| **Chen et al. (2023)** — Position Interpolation (arXiv:2306.15595) | `src/positional.py` references it; paper mentions PI in Section 4.4 but never cites it. | Add to `references.bib` and cite in Section 4.4. |
| **Snover et al. (2006)** — TER metric | TER reported in Table 1, never cited. | Add to `references.bib` and cite when TER is first introduced. |
| **Liu et al. (2022)** — xPos | Related work on RoPE variants is incomplete without xPos. | Add to `references.bib` and discuss in Related Work. |
| **WALS / Dryer & Haspelmath** | "Hindi free word order," "Bengali agglutinative" — uncited linguistic claims. | Add WALS citations OR remove unsupported typological claims. |
| **MT-specific RoPE usage** | Paper claims RoPE "dominates MT" but only cites LLMs (LLaMA, Mistral). | Either cite actual MT systems using RoPE (e.g., NLLB papers if applicable) OR rephrase to "dominates large language models." |
| **Xiao et al. (or similar)** — ALiBi sink-ratio property | "A known ALiBi property" (sink ratios) is uncited. Press et al. does not discuss this. | Find and cite the correct source for ALiBi sink-ratio analysis. |

**Files:** `paper/references.bib`, `paper/main.tex`

---

### 6. Overclaims and Unsupported Statements

| Statement | Issue | Fix |
|-----------|-------|-----|
| *"RoPE has become the dominant positional encoding in neural machine translation"* | RoPE dominates in **LLMs**, not publicly documented MT systems. NLLB, DeepL, Google Translate do not disclose RoPE. | Rephrase: *"RoPE has become dominant in large language models and is increasingly adopted in sequence-to-sequence architectures."* |
| *"confirming a systematic low-frequency preference"* | Gates converge below 1, but this is uniform suppression across all frequencies. Not proven "low-frequency" preference. | Rephrase: *"confirming systematic frequency suppression"* or analyze per-frequency scaling to justify "low-frequency." |
| *"Hindi's rich fusional morphology and relatively free word order benefit from phase offsets"* | Pure speculation. No causal evidence links phase offsets to morphology or word order. | Either (a) add mechanistic analysis showing how phase offsets affect attention to morphological markers, or (b) rephrase as *"We hypothesize that..."* and acknowledge it is post-hoc. |
| *"Bengali's more regular word order and agglutinative morphology benefit from frequency scaling"* | Bengali is not straightforwardly "agglutinative." Uncited typological claim. | Fix the typological description (Bengali is fusional, not agglutinative) and rephrase as hypothesis. |
| *"This reveals a fundamental accuracy–extrapolation trade-off"* | Observed at 47M parameters with greedy decoding. Not demonstrated to be "fundamental" or generalizable. | Rephrase: *"This suggests an accuracy–extrapolation trade-off in our experimental setting."* |
| *"To our knowledge, no prior work systematically compares PE requirements across morphologically diverse language pairs with controlled architecture and training"* | Training is explicitly **not controlled** across language pairs. | Either control training (Issue #2) OR remove this uniqueness claim. |

**Files:** `paper/main.tex`

---

### 7. Statistical Rigor Issues

**Problem:**  
- No significance tests for Hi-En / Bn-En. Table 2 (`tab:sig`) only covers En-De.
- Welch's t-test on RoPE (20.83 ± 0.03) vs AdaptiveRoPE (20.69 ± 0.05) reports p=0.001. With such tiny variance, significance is trivial but **practically meaningless** (0.14 BLEU).
- Bootstrap resampling mentioned in Section 4.3 but never reported.

**Fix:**  
- Add significance tests (bootstrap or paired t-test) for Hi-En and Bn-En after multi-seed training (Issue #3).
- Report **effect sizes** (Cohen's d) alongside p-values for En-De.
- Either report bootstrap confidence intervals in a table/appendix OR remove the mention of bootstrap resampling.

**Files:** `paper/main.tex`, `pipeline/evaluate_model.py` or analysis scripts

---

### 8. Length Generalization Table Lacks Absolute Values
**Problem:**  
Table 4 shows relative % degradation (e.g., RoPE −97% at 201–400 words) but never shows **absolute BLEU per bucket**. A −97% drop from 20.8 to 0.6 is very different from 5.0 to 0.15.

**Fix:**  
Add a supplementary table or appendix showing absolute BLEU per length bucket for each method.

**Files:** `paper/main.tex`

---

## 🟢 MINOR — Polish Before Submission

### 9. Citation Key Inconsistencies
- `\cite{loshchilov2018decoupled}` is used but the paper is from **2019** (`.bib` correctly lists 2019). The key name is misleading.
- `\cite{press2021train}` key says 2021 but the paper is **ICLR 2022** (`.bib` correctly lists 2022).

**Fix:**  
Update citation keys to match actual publication years: `loshchilov2019decoupled`, `press2022train`. Update all `\cite{}` calls accordingly.

**Files:** `paper/references.bib`, `paper/main.tex`

---

### 10. Placeholder in Author Block
**Problem:**  
`[Your Institution]` placeholder remains in the author block.

**Fix:**  
Replace with actual affiliation or use `\author{}` with anonymized institution for blind review.

**Files:** `paper/main.tex`

---

### 11. Table 1 Caption Inconsistency
**Problem:**  
Caption says "mean ± std over **2–3 seeds**" but text says 3 seeds (42, 43, 44). The "2" is unexplained.

**Fix:**  
Clarify: if one seed failed, explain why. Otherwise, say "3 seeds."

**Files:** `paper/main.tex`

---

### 12. Inconsistent Percentage Claims
**Problem:**  
Abstract says "+141% BLEU" for Sinusoidal vs RoPE on Hi-En. Later text says "+21% BLEU" for PhasesOnly vs RoPE on Hi-En. Both are correct but refer to different comparisons, which is confusing.

**Fix:**  
Always specify the baseline when reporting relative improvement, e.g., *"Sinusoidal outperforms RoPE by +141% BLEU (18.80 vs 7.78), while PhasesOnly improves over RoPE by +21% (9.42 vs 7.78)."*

**Files:** `paper/main.tex`

---

### 13. Greedy Decoding Not Justified
**Problem:**  
The paper uses greedy decoding for all results. Most MT systems use beam search (beam=4–5). Greedy is faster but lower quality. The choice is not explicitly justified.

**Fix:**  
Add a sentence justifying greedy decoding (e.g., "We use greedy decoding for speed and fair comparison; Appendix X shows beam=5 results follow the same ranking"). Optionally, add a small beam search comparison in the appendix.

**Files:** `paper/main.tex`

---

## Summary Priority List

| Priority | Issue | Effort | Impact on Submission |
|----------|-------|--------|----------------------|
| **P0** | #1 Cross-attention PE asymmetry | High (recode + retrain) | **Fatal if not fixed** |
| **P0** | #2 Uncontrolled training across languages | High (retrain) or Low (rewrite claims) | Invalidates cross-lingual claims |
| **P0** | #3 Single-seed for Indic languages | High (retrain) | No statistical validity |
| **P1** | #4 Low absolute BLEU scores | Medium (debug + retrain) | Undermines all claims |
| **P1** | #5 Missing citations | Low | Weakens related work |
| **P1** | #6 Overclaims/unsupported statements | Low | Damages credibility |
| **P2** | #7 Statistical rigor | Medium | Needed for top-tier venue |
| **P2** | #8 Length gen absolute values | Low | Improves transparency |
| **P3** | #9–13 Minor polish | Very Low | Submission readiness |

---

## Recommended Action Plan

### Phase 1: Fix the Architecture (1–2 days)
1. Implement Option A from Issue #1: add RoPE to cross-attention K in `MultiHeadCrossAttention`.
2. Verify Sinusoidal still works (should be unchanged).
3. Run a quick sanity check: train RoPE on Hi-En for 5K steps and verify BLEU > 10 (if not, there are deeper issues).

### Phase 2: Re-train Everything (1–2 weeks)
1. Decide on unified hyperparameters (Issue #2) — or commit to per-language analysis only.
2. Train ALL models with **3 seeds** (Issue #3).
3. Evaluate all models with greedy + beam search.

### Phase 3: Rewrite Paper (2–3 days)
1. Update Table 1 with new results.
2. Rewrite abstract, introduction, and discussion to reflect fair comparison.
3. Remove or rephrase unsupported linguistic claims (Issue #6).
4. Add missing citations (Issue #5).
5. Add significance tests for all language pairs (Issue #7).
6. Fix minor issues (#9–13).

### Phase 4: Internal Review (1 day)
1. Have a colleague read the paper focusing on: (a) fairness of comparison, (b) strength of claims vs evidence.
2. Re-compile PDF and verify all citations resolve.

---

*Last updated: 2026-05-21*
