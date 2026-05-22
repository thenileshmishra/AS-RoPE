#!/bin/bash
# Run all missing training jobs with CONSISTENT hyperparameters per language.
#
# En-De: seq=128 batch=256 steps=25K lr=0.001 (matches rope_de / asrope3_de)
# Hi-En: seq=192 batch=64  steps=75K lr=0.0005 (matches majority of Hi-En runs)
# Bn-En: seq=192 batch=64  steps=75K lr=0.0005 (matches all Bn-En runs)

cd "$(dirname "$0")"
source .venv/bin/activate

run_if_missing() {
    run_name="$1"
    shift
    if [ -f "outputs/logs/${run_name}/run_summary.json" ]; then
        echo "[SKIP] ${run_name} already complete"
        return 0
    fi
    echo "[RUN]  ${run_name}"
    "$@"
}

# ═══════════════════════════════════════════════════════════════════════════════
# En-De  (seq=128  batch=256  steps=25K  lr=0.001)
# ═══════════════════════════════════════════════════════════════════════════════

run_if_missing rope_de_s43 \
python -m pipeline.train_model \
    --pe-type rope --run-name rope_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

run_if_missing rope_de_s44 \
python -m pipeline.train_model \
    --pe-type rope --run-name rope_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

run_if_missing adaptiverope_de_s43 \
python -m pipeline.train_model \
    --pe-type adaptiverope --run-name adaptiverope_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

run_if_missing adaptiverope_de_s44 \
python -m pipeline.train_model \
    --pe-type adaptiverope --run-name adaptiverope_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

run_if_missing sinusoidal_de_s43 \
python -m pipeline.train_model \
    --pe-type sinusoidal --run-name sinusoidal_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

run_if_missing sinusoidal_de_s44 \
python -m pipeline.train_model \
    --pe-type sinusoidal --run-name sinusoidal_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

run_if_missing gatesonly_de_s42 \
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_de_s42 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

run_if_missing gatesonly_de_s43 \
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

run_if_missing gatesonly_de_s44 \
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

run_if_missing phasesonly_de_s42 \
python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_de_s42 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

run_if_missing phasesonly_de_s43 \
python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

run_if_missing phasesonly_de_s44 \
python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --max-seq-len 128 --batch-size 256 --num-steps 25000 \
    --eval-every 1000 --learning-rate 1e-3 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

# ═══════════════════════════════════════════════════════════════════════════════
# Hi-En  (seq=192  batch=64  steps=75K  lr=0.0005)
# ═══════════════════════════════════════════════════════════════════════════════

run_if_missing gatesonly_hi_s42 \
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_hi_s42 \
    --tokenized-train processed_data/tokenized/train_1m_hi_en.pt \
    --tokenized-val processed_data/tokenized/val_hi_en.pt \
    --max-seq-len 192 --batch-size 64 --num-steps 75000 \
    --eval-every 1000 --learning-rate 5e-4 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

run_if_missing phasesonly_hi_s42 \
python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_hi_s42 \
    --tokenized-train processed_data/tokenized/train_1m_hi_en.pt \
    --tokenized-val processed_data/tokenized/val_hi_en.pt \
    --max-seq-len 192 --batch-size 64 --num-steps 75000 \
    --eval-every 1000 --learning-rate 5e-4 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

# ═══════════════════════════════════════════════════════════════════════════════
# Bn-En  (seq=192  batch=64  steps=75K  lr=0.0005)
# ═══════════════════════════════════════════════════════════════════════════════

run_if_missing sinusoidal_bn_s42 \
python -m pipeline.train_model \
    --pe-type sinusoidal --run-name sinusoidal_bn_s42 \
    --tokenized-train processed_data_bn/tokenized/train_bn_en.pt \
    --tokenized-val processed_data_bn/tokenized/val_bn_en.pt \
    --max-seq-len 192 --batch-size 64 --num-steps 75000 \
    --eval-every 1000 --learning-rate 5e-4 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

run_if_missing gatesonly_bn_s42 \
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_bn_s42 \
    --tokenized-train processed_data_bn/tokenized/train_bn_en.pt \
    --tokenized-val processed_data_bn/tokenized/val_bn_en.pt \
    --max-seq-len 192 --batch-size 64 --num-steps 75000 \
    --eval-every 1000 --learning-rate 5e-4 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

run_if_missing phasesonly_bn_s42 \
python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_bn_s42 \
    --tokenized-train processed_data_bn/tokenized/train_bn_en.pt \
    --tokenized-val processed_data_bn/tokenized/val_bn_en.pt \
    --max-seq-len 192 --batch-size 64 --num-steps 75000 \
    --eval-every 1000 --learning-rate 5e-4 --grad-accum 1 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

echo "All training jobs complete!"
