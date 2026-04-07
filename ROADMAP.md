# 15-Day Single Roadmap

## 1) Target Domain
Low-resource multilingual machine translation for Indian languages, with Hindi-English as the main pair and your positional encoding work as the core research angle.

Why this domain:
- Your repo already has strong positional encoding implementations.
- MT gives real-world impact and publishable evaluation metrics.
- Hindi-English is practical in 15 days and still meaningful.

Encoding baseline rule:
- Add classic sinusoidal positional encoding as one reference baseline.
- Keep the rest to the encodings already in the repo: RoPE, ALiBi, NTK-Scaled RoPE, and your own lightweight variant.
- Do not add any more encodings.

Language-pair rule:
- Use Hindi-English only for the 15-day plan.
- Do not add a second pair unless the Hindi-English pipeline is fully working by Day 7.

## 2) What To Improve In Current Repo
Current repo is good at language modeling + synthetic retrieval, but not yet a translation research project.

You must improve these exact points:

1. Add real MT data pipeline
- Create a parallel dataset loader for Hindi-English.
- Add clean train/val/test split.

2. Add translation task training loop
- Keep your transformer backbone.
- Train for seq2seq-style next-token generation on source+target format.

3. Add MT evaluation metrics
- BLEU, chrF, and exact reproducible eval script.

4. Compare positional encodings on the same MT task
- RoPE vs ALiBi vs NTK-Scaled RoPE.
- Same training budget for fair comparison.

5. Add one clear contribution
- A small language-aware positional scaling variant (simple, not huge redesign).
- Show it improves at least one metric over best baseline.

6. Tight experiment hygiene
- Fixed seeds.
- One config folder.
- One results table generated automatically.

## 3) Your Goal
In 15 days, produce one focused research artifact:

Goal statement:
"Demonstrate, on low-resource Hindi-English MT, that positional encoding choice materially affects translation quality, and that a lightweight language-aware scaling variant can improve over standard baselines under fixed compute."

Hard deliverables by Day 15:
- Working MT training pipeline.
- Baseline results for RoPE, ALiBi, NTK-Scaled.
- Your variant result.
- One final table + one figure.
- 4-6 page draft report (problem, method, setup, results, error analysis).

---

## Concrete 15-Day Plan (14 hours/day)

Daily structure (repeat every day):
- Block A (5h): implementation
- Block B (5h): training/eval runs
- Block C (4h): analysis + writing + cleanup

### Day 1
- Freeze scope to Hindi-English only.
- Create branch and clean run scripts.
- Add `data/mt_dataset.py` for parallel sentence loading.
- Output: first batch loads correctly.

### Day 2
- Add preprocessing pipeline:
  - normalize text
  - tokenize with one tokenizer strategy
  - deterministic train/val/test split
- Output: cached tokenized dataset files.

### Day 3
- Implement MT training entrypoint `src/train_mt.py`.
- Reuse existing model and positional encoding switch.
- Output: 1 short smoke training run completes.

### Day 4
- Implement generation + decode script `src/decode_mt.py`.
- Implement eval script `src/eval_mt.py` with BLEU + chrF.
- Output: end-to-end train -> decode -> metrics works.

### Day 5
- Baseline run 1: RoPE.
- Save checkpoint + metrics JSON.
- Output: first real baseline number.

### Day 6
- Baseline run 2: ALiBi.
- Same compute budget as Day 5.
- Output: second baseline number.

### Day 7
- Baseline run 3: NTK-Scaled RoPE.
- Same compute budget.
- Output: third baseline number.

### Day 8
- Build unified result aggregator script:
  - reads all metrics JSONs
  - writes one CSV + one latex/plain table
- Output: single baseline comparison table.

### Day 9
- Implement your variant:
  - lightweight language-aware scaling factor over rotary frequencies
  - minimal code delta only
- Output: code compiles + smoke run passes.

### Day 10
- Full run of your variant under same budget.
- Output: comparable metric row added.

### Day 11
- Run ablation A:
  - variant without learned scaling (fixed scaling)
- Run ablation B:
  - variant with learned scaling
- Output: ablation mini-table.

### Day 12
- Error analysis on 100 sampled translations:
  - named entities
  - word order
  - morphology
- Output: short qualitative findings section.

### Day 13
- Make one final figure:
  - bar chart BLEU/chrF for all models
- Start writing report sections 1-3 (intro, method, setup).

### Day 14
- Write report sections 4-6 (results, analysis, limitations).
- Re-run only missing experiments if gaps exist.

### Day 15
- Final consistency pass:
  - check seeds
  - check table values
  - check commands in reproducibility section
- Final outputs packed:
  - report PDF/tex
  - results table CSV
  - figure PNG

---

## Non-Negotiable Rules (Keep Focus)
- One language pair only: Hindi-English.
- One primary metric to optimize: BLEU (track chrF as secondary).
- No architecture overhaul.
- No extra domains.
- No new side experiments unless a core run fails.
- One extra baseline only: sinusoidal positional encoding.

## Definition of Done
Done means all 6 are true:
1. `train_mt.py` runs end-to-end.
2. `eval_mt.py` outputs BLEU and chrF.
3. RoPE/ALiBi/NTK results exist.
4. Your variant result exists.
5. One final comparison table + one figure exist.
6. One short research draft is complete.