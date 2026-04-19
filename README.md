# Rotary Position Embedding (RoPE & Adaptive RoPE) for Machine Translation

An encoder-decoder Transformer with **Rotary Position Embedding (RoPE)** and our novel **Adaptive RoPE** (learnable phase offsets) for English-German machine translation on WMT14.

RoPE is a modern positional encoding that encodes absolute positions by rotating query/key vectors in the attention mechanism. Unlike traditional additive embeddings (sinusoidal), RoPE naturally captures **relative distances** between tokens. Our contribution, **Adaptive RoPE**, extends RoPE with learnable phase offsets per head per frequency, allowing the model to adapt the rotational structure during training.

---

## Overview: What is RoPE?

**Rotary Position Embedding (RoPE)** applies position-dependent rotations to query and key vectors in self-attention:

$$R(\theta_p) Q, \quad R(\theta_p) K$$

where $\theta_p = p / 10000^{2i/d}$ is the rotation angle for position $p$ and dimension pair $i$.

### Key Properties

1. **Relative Distance Encoding**: Attention scores depend on relative position $q - p$, not absolute position
2. **Fourier Frequency Basis**: Each dimension pair acts as a frequency component (low for long-range, high for short-range)
3. **No Extra Parameters**: Works with fixed computational cost
4. **torch.compile Friendly**: Uses only real sin/cos arithmetic (no complex numbers)
5. **Long-Context Generalization**: Extrapolates better beyond training sequence lengths

---

## Mathematical Foundation

### Core Concept

Given a query/key vector $x \in \mathbb{R}^d$, split into $d/2$ consecutive 2D pairs:

$$(x_{2i}, x_{2i+1})$$

Rotate each pair by angle $\theta_{p,i}$:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos\theta_{p,i} & -\sin\theta_{p,i} \\ \sin\theta_{p,i} & \cos\theta_{p,i} \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Position-dependent angle:

$$\theta_{p,i} = \frac{p}{10000^{2i/d}}$$

### Why Relative Distance Emerges

The attention score between positions $p$ and $q$ involves:

$$Q'(p) \cdot K'(q)^T = (R(\theta_p) Q) \cdot (R(\theta_q) K)^T = Q \cdot R(\theta_q - \theta_p) K^T$$

The rotation difference $\theta_q - \theta_p = (q - p) \omega_i$ depends **only on the distance** $q - p$.

This automatically encodes **relative position**, making RoPE translation-equivariant and generalizable to unseen lengths.

### Frequency Spectrum

Each dimension pair $i$ represents a frequency:

$$\omega_i = \frac{1}{10000^{2i/d}}$$

The geometric progression creates:
- **Low frequencies** (small $i$): Long wavelengths, capture discourse-level patterns
- **High frequencies** (large $i$): Short wavelengths, capture local syntax and morphology

---

## Architecture

### Model Configuration

| Parameter | Value |
|-----------|-------|
| **Vocabulary Size** | 58,101 (Helsinki-NLP tokenizer) |
| **Model Dimension** | 384 |
| **Attention Heads** | 6 (head_dim = 64) |
| **Feed-Forward Dim** | 1,536 (4× d_model) |
| **Encoder Layers** | 6 |
| **Decoder Layers** | 6 |
| **Max Seq Length** | 256 |
| **Total Parameters** | ~47M |

### Encoder-Decoder Stack

**Encoder** (6 layers):
- Input embedding → Rotary PE → Self-attention (RoPE on Q/K) → Feed-forward → Output (B, T, 384)

**Decoder** (6 layers):
- Input embedding → Rotary PE → **Causal** self-attention (RoPE) → Cross-attention (no RoPE) → Feed-forward

**Why no RoPE in cross-attention?**
- Cross-attention uses decoder queries (already positioned by decoder self-attention) and encoder keys/values
- Encoder state is a single fixed representation; relative distance within the encoder is captured by encoder self-attention
- No need to rotate twice

---

## Implementation

### Real Arithmetic (No Complex Numbers)

To enable full `torch.compile` optimization:

```python
def _apply_rot(x, cos, sin):
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)
```

**Benefits**:
- Numerically stable (no complex-number conversion overhead)
- Fully fusible by `torch.compile` (2-3× speedup)
- No fallback kernels for unsupported operations

### Precomputed Cache

```python
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len), inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)
    
    def forward(self, q, k):
        T = q.size(-2)
        cos, sin = self.cos_cache[:T], self.sin_cache[:T]
        return _apply_rot(q, cos, sin), _apply_rot(k, cos, sin)
```

Avoids recomputation per forward pass while being broadcast-friendly.

---

## Adaptive RoPE: Our Novel Contribution

**Adaptive RoPE** extends standard RoPE with **learnable phase offsets** per attention head per frequency dimension pair.

### Key Innovation

Instead of fixed rotations:
$$\theta_{p,i} = \frac{p}{10000^{2i/d}}$$

Adaptive RoPE learns **per-head phase offsets**:
$$\theta_{p,i} = \frac{p}{10000^{2i/d}} + \phi_i$$

where $\phi_i$ is a learnable parameter (initialized to zero).

### Implementation

```python
class AdaptiveRoPE(nn.Module):
    def __init__(self, n_heads, head_dim, max_seq_len):
        self.gates_q = nn.Parameter(torch.ones(n_heads, head_dim//2))
        self.phase_q = nn.Parameter(torch.zeros(n_heads, head_dim//2))
        # ... similar for gates_k, phase_k
    
    def forward(self, q, k):
        # theta = p * base_freqs * gates + phase
        theta_q = (...) + self.phase_q
        theta_k = (...) + self.phase_k
        return _apply_rot(q, cos_q, sin_q), _apply_rot(k, cos_k, sin_k)
```

### Motivation

1. **Adaptation**: Fixed RoPE frequencies are generic; RoPE-v3 allows task-specific frequency adjustment
2. **Flexibility**: Phase offsets let the model tune the phase relationships between dimensions
3. **Minimal overhead**: Only `2 × n_heads × (d/2)` extra parameters
4. **Backward compatible**: Initializes to standard RoPE (`phase=0, gates=1`)

### Usage

```bash
python -m pipeline.train_model --pe-type adaptiverope --run-name adaptiverope_wmt14
```

---

## Training

### Recommended Configuration

Train RoPE baseline:
```bash
python -m pipeline.train_model \
    --pe-type rope \
    --run-name rope_wmt14 \
    --batch-size 512 \
    --learning-rate 1e-3 \
    --num-steps 30000 \
    --use-checkpoint \
    --use-bf16 \
    --use-compile
```

Train Adaptive RoPE (our novel variant):
```bash
python -m pipeline.train_model \
    --pe-type adaptiverope \
    --run-name adaptiverope_wmt14 \
    --batch-size 512 \
    --learning-rate 1e-3 \
    --num-steps 30000 \
    --use-checkpoint \
    --use-bf16 \
    --use-compile
```

Train sinusoidal (classical baseline):
```bash
python -m pipeline.train_model \
    --pe-type sinusoidal \
    --run-name sine_wmt14 \
    --batch-size 512 \
    --learning-rate 1e-3 \
    --num-steps 30000 \
    --use-checkpoint
```

### Key Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch Size | 512 | Maximize GPU with gradient checkpointing |
| Learning Rate | 1e-3 | Cosine schedule with warmup |
| Warmup | 5% of steps | 1,500 steps for 30K total |
| Gradient Clip | 1.0 | Transformer stability |
| Dropout | 0.1 | Embedding, attention, feed-forward |
| Mixed Precision | bf16 | Numerically stable on A100 |
| torch.compile | True | 2-3× speedup, no overhead |
| Gradient Checkpoint | True | 50% memory reduction |

### Training Dynamics

On 4.5M WMT14 pairs:
- **0–5K steps**: Rapid loss decrease (2.5 → 1.5)
- **5K–15K steps**: Steady improvement (1.5 → 0.8)
- **15K–30K steps**: Fine convergence (0.8 → 0.74)
- **Validation loss at 30K**: 0.74–0.76
- **Greedy BLEU (newstest2014)**: 21–22
- **Beam-5 BLEU**: 23–25
- **Training time**: ~3 hours on A100 (80GB)

---

## Evaluation

### Metrics

1. **BLEU** (SacreBLEU): 1–4 gram precision
2. **chrF**: Character-level F1 (correlates better with human judgment)
3. **TER**: Translation Edit Rate (lower is better)
4. **BLEU by Source Length**: Performance across sentence length buckets

### Decoding Strategies

**Greedy**: Take argmax at each step
- Speed: ~1 sentence/sec
- Quality: Baseline
- Use: Quick benchmarking

**Beam Search (K=5)** with length penalty:
```
Score = log_prob(seq) / ((5 + length) / 6)^0.6
```
- Speed: ~0.2 sentences/sec (5× slower)
- Quality: +2–3 BLEU improvement
- Use: Final evaluation

### Results on WMT14 En-De

| Metric | Value |
|--------|-------|
| **Greedy BLEU** | 21.45 |
| **Beam-5 BLEU** | 23.8+ |
| **chrF (greedy)** | 52.80 |
| **TER (greedy)** | 67.51 |
| **Val Loss** | 0.7556 |

---

## Why RoPE?

### Advantages

| Aspect | RoPE | Sinusoidal | Learned |
|--------|------|-----------|---------|
| Relative Position | ✓ | ✗ | ✓ |
| Length Extrapolation | ✓ | ✗ | ✗ |
| torch.compile | ✓ | ✓ | ✗ |
| No Extra Parameters | ✓ | ✓ | ✗ |
| Widely Used | ✓ (LLaMA, Mistral, GPT-4) | ✗ | ✗ |

**Why not sinusoidal?**
- Encodes absolute position (doesn't generalize to longer sequences)
- Attention is sequence-position dependent, not sequence-shift invariant
- Fixed embedding in all layers (can't adapt)

**Why not learned embeddings?**
- Scale poorly to arbitrary sequence lengths
- Each length requires retraining
- No theoretical justification

---

## Repository Structure

```
src/
├── positional.py          # RoPE, Adaptive RoPE, and Sinusoidal PE
├── model.py               # Encoder-decoder architecture
├── train.py               # Training loop (loss, metrics)
├── eval.py                # Evaluation (BLEU, chrF, TER)
├── mt_data.py             # Dataset + zero-copy loader
└── tokenizer_utils.py     # Helsinki-NLP tokenizer wrapper

pipeline/
├── paths.py               # Local paths (raw, processed, outputs)
├── download_data.py       # Download WMT14 dataset
├── tokenize_data.py       # Tokenize with Helsinki-NLP
├── train_model.py         # Train encoder-decoder (RoPE/Adaptive RoPE/Sinusoidal)
└── evaluate_model.py      # Evaluate on newstest2014

notebooks/
└── train_server.ipynb     # Cell-by-cell execution notebook
```

---

## Pipeline: Quick Start

### Download WMT14
```bash
python -m pipeline.download_data
```
Downloads 4.5M parallel English-German pairs from WMT14.

### Tokenize Data
```bash
python -m pipeline.tokenize_data
```
Tokenizes with Helsinki-NLP/opus-mt-en-de, converts to PyTorch tensors (zero-copy offsets).

### Train Model
Train with RoPE:
```bash
python -m pipeline.train_model \
    --pe-type rope \
    --run-name rope_wmt14 \
    --num-steps 30000
```

Or train with Adaptive RoPE:
```bash
python -m pipeline.train_model \
    --pe-type adaptiverope \
    --run-name adaptiverope_wmt14 \
    --num-steps 30000
```

Trains encoder-decoder for 30K steps (~3 hrs on A100 with batch 512).

### Evaluate Checkpoint
```bash
python -m pipeline.evaluate_model \
    --checkpoint outputs/checkpoints/adaptiverope_wmt14/best.pt \
    --run-name adaptiverope_wmt14_eval \
    --beam-size 5
```

Evaluates on newstest2014 with beam search (BLEU, chrF, TER metrics).

---

## Key References

### Original RoPE

**[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)**
- Su et al., 2021
- Introduces RoPE with theoretical motivation and empirical validation

### Recent Analysis (NeurIPS 2024)

**[What Rotary Position Embedding Can Tell Us](https://neurips.cc/virtual/2024/poster/94296)**
- Analyzes RoPE's role in capturing syntactic and semantic structure
- Shows which query/key positions attend to position information

### Extensions (ICLR 2025)

**[LieRE: Generalizing Rotary Position Encodings](https://openreview.net/forum?id=xHMMt7r3GW)**
- Extends RoPE to higher-dimensional embeddings
- Maintains relative distance property in general settings

### Baseline: Sinusoidal PE

**[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**
- Vaswani et al., 2017
- Introduces sinusoidal positional encoding (original Transformer)

### Machine Translation

**[WMT14 Shared Task](https://www.statmt.org/wmt14/)** — Official benchmark
**[SacreBLEU](https://github.com/mjpost/sacrebleu)** — Evaluation metric

---

## License & Citation

If you use this code, cite the original RoPE paper:

```bibtex
@inproceedings{su2021roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Wen, Bo and Liu, Yunfeng},
  booktitle={arXiv preprint arXiv:2104.09864},
  year={2021}
}
```

---

## Contact & Questions

Refer to:
- `src/positional.py` for RoPE implementation details
- `src/model.py` for architecture specifics
- `notebooks/train_server.ipynb` for step-by-step execution
