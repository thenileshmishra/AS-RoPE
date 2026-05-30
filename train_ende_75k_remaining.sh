#!/bin/bash
# Resume En-De 75K training: ALiBi s42 (restart), RoPE s43, RoPE s44

set -e

COMMON_ARGS="--tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt --num-steps 75000 --eval-every 1000 --batch-size 256 --learning-rate 1e-3 --max-seq-len 128 --grad-accum 2 --use-bf16 --use-compile"

# 1. Restart ALiBi s42 (was killed at step 34K)
echo "=== Training ALiBi En-De seed 42 (75K) ==="
python -m pipeline.train_model --run-name alibi_de_s42_75k --pe-type alibi $COMMON_ARGS --seed 42

# 2. RoPE s43 for baseline variance
echo "=== Training RoPE En-De seed 43 (75K) ==="
python -m pipeline.train_model --run-name rope_de_s43_75k --pe-type rope $COMMON_ARGS --seed 43

# 3. RoPE s44 for baseline variance
echo "=== Training RoPE En-De seed 44 (75K) ==="
python -m pipeline.train_model --run-name rope_de_s44_75k --pe-type rope $COMMON_ARGS --seed 44

echo "=== ALL REMAINING TRAINING COMPLETE ==="
