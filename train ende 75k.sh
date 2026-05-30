#!/bin/bash
# Train all En-De models at 75K steps for consistent comparison
# Run sequentially on single GPU

set -e

COMMON_ARGS="--tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt --num-steps 75000 --eval-every 1000 --batch-size 256 --learning-rate 1e-3 --max-seq-len 128 --grad-accum 2 --use-bf16 --use-compile"

# Seed 42 for all methods
echo "=== Training RoPE En-De seed 42 (75K) ==="
python -m pipeline.train_model --run-name rope_de_s42_75k --pe-type rope $COMMON_ARGS --seed 42

echo "=== Training AdaptiveRoPE En-De seed 42 (75K) ==="
python -m pipeline.train_model --run-name adaptiverope_de_s42_75k --pe-type adaptiverope $COMMON_ARGS --seed 42

echo "=== Training GatesOnly En-De seed 42 (75K) ==="
python -m pipeline.train_model --run-name gatesonly_de_s42_75k --pe-type gatesonly $COMMON_ARGS --seed 42

echo "=== Training PhasesOnly En-De seed 42 (75K) ==="
python -m pipeline.train_model --run-name phasesonly_de_s42_75k --pe-type phasesonly $COMMON_ARGS --seed 42

echo "=== Training ALiBi En-De seed 42 (75K) ==="
python -m pipeline.train_model --run-name alibi_de_s42_75k --pe-type alibi $COMMON_ARGS --seed 42

# Optional: seeds 43, 44 for RoPE only (baseline variance)
echo "=== Training RoPE En-De seed 43 (75K) ==="
python -m pipeline.train_model --run-name rope_de_s43_75k --pe-type rope $COMMON_ARGS --seed 43

echo "=== Training RoPE En-De seed 44 (75K) ==="
python -m pipeline.train_model --run-name rope_de_s44_75k --pe-type rope $COMMON_ARGS --seed 44

echo "=== ALL TRAINING COMPLETE ==="
