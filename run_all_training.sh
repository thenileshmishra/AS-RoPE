#!/bin/bash
# Run all missing training jobs sequentially
# This script will take ~2 days to complete

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

# En-De: additional seeds for existing methods
python -m pipeline.train_model \
    --pe-type rope --run-name rope_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

python -m pipeline.train_model \
    --pe-type rope --run-name rope_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

python -m pipeline.train_model \
    --pe-type adaptiverope --run-name adaptiverope_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

python -m pipeline.train_model \
    --pe-type adaptiverope --run-name adaptiverope_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

python -m pipeline.train_model \
    --pe-type sinusoidal --run-name sinusoidal_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

python -m pipeline.train_model \
    --pe-type sinusoidal --run-name sinusoidal_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

# En-De: new ablations
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_de_s42 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_de_s42 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_de_s43 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 43

python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_de_s44 \
    --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
    --tokenized-val processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 44

# Hi-En: new ablations
python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_hi_s42 \
    --tokenized-train processed_data/tokenized/train_1m_hi_en.pt \
    --tokenized-val processed_data/tokenized/val_hi_en.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_hi_s42 \
    --tokenized-train processed_data/tokenized/train_1m_hi_en.pt \
    --tokenized-val processed_data/tokenized/val_hi_en.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

# Bn-En: missing baselines
python -m pipeline.train_model \
    --pe-type sinusoidal --run-name sinusoidal_bn_s42 \
    --tokenized-train processed_data_bn/tokenized/train_bn_en.pt \
    --tokenized-val processed_data_bn/tokenized/val_bn_en.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

python -m pipeline.train_model \
    --pe-type gatesonly --run-name gatesonly_bn_s42 \
    --tokenized-train processed_data_bn/tokenized/train_bn_en.pt \
    --tokenized-val processed_data_bn/tokenized/val_bn_en.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

python -m pipeline.train_model \
    --pe-type phasesonly --run-name phasesonly_bn_s42 \
    --tokenized-train processed_data_bn/tokenized/train_bn_en.pt \
    --tokenized-val processed_data_bn/tokenized/val_bn_en.pt \
    --num-steps 25000 --eval-every 1000 --batch-size 512 --learning-rate 1e-3 \
    --use-checkpoint --use-bf16 --use-compile --seed 42

echo "All training jobs complete!"
