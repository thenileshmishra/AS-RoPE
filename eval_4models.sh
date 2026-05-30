#!/bin/bash
# Evaluate 4 completed En-De 75K models on WMT14 test
set -e

# Common args
EVAL_TSV="raw_data/wmt14/test.tsv"
BATCH=32
BEAM=5

echo "=== Evaluating RoPE s42 (75K) ==="
python -m pipeline.evaluate_model \
    --checkpoint outputs/checkpoints/rope_de_s42_75k/best.pt \
    --run-name rope_de_s42_75k_eval \
    --eval-tsv "$EVAL_TSV" --batch-size $BATCH --beam-size $BEAM

echo "=== Evaluating AdaptiveRoPE s42 (75K) ==="
python -m pipeline.evaluate_model \
    --checkpoint outputs/checkpoints/adaptiverope_de_s42_75k/best.pt \
    --run-name adaptiverope_de_s42_75k_eval \
    --eval-tsv "$EVAL_TSV" --batch-size $BATCH --beam-size $BEAM

echo "=== Evaluating GatesOnly s42 (75K) ==="
python -m pipeline.evaluate_model \
    --checkpoint outputs/checkpoints/gatesonly_de_s42_75k/best.pt \
    --run-name gatesonly_de_s42_75k_eval \
    --eval-tsv "$EVAL_TSV" --batch-size $BATCH --beam-size $BEAM

echo "=== Evaluating PhasesOnly s42 (75K) ==="
python -m pipeline.evaluate_model \
    --checkpoint outputs/checkpoints/phasesonly_de_s42_75k/best.pt \
    --run-name phasesonly_de_s42_75k_eval \
    --eval-tsv "$EVAL_TSV" --batch-size $BATCH --beam-size $BEAM

echo "=== ALL EVALUATIONS COMPLETE ==="
