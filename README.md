# Neur — Hindi ↔ English MT (sinusoidal baseline)

A minimal, sequential Colab + Google Drive pipeline for training a decoder-only
transformer with sinusoidal positional encoding on Samanantar and evaluating it
on FLORES-200.

## Layout

```
pipeline/                 # Sequential Colab steps
  paths.py                #   Google Drive paths (override via NEUR_DRIVE_ROOT)
  step1_download.py       #   Step 1: download raw datasets → Drive/raw_data
  step2_preprocess.py     #   Step 2: clean + split → Drive/processed_data
  step3_train.py          #   Step 3: train (sinusoidal) + evaluate → Drive/outputs
src/                      # Core library
  model.py                #   GPT with sinusoidal positional encoding
  sinusoidal.py           #   Sinusoidal PE module
  dataset.py              #   Parallel TSV dataset + collate
  train.py                #   Training loop
  eval.py                 #   Greedy decode + BLEU/chrF
notebooks/
  pipeline.ipynb          # Colab notebook with one cell per step
```

## Google Drive layout

All artifacts live under a single Drive folder (default
`/content/drive/MyDrive/neur`, override with `NEUR_DRIVE_ROOT`):

```
neur/
  raw_data/
    samanantar/samanantar_hi_en.tsv
    flores200/flores200_hi_en_devtest.tsv
  processed_data/
    train.tsv  val.tsv  test.tsv  metadata.json
  outputs/
    checkpoints/best.pt  checkpoints/last.pt
    logs/metrics.jsonl  logs/run_config.json  logs/run_summary.json
    metrics/metrics.json  metrics/pred.txt  metrics/ref.txt  metrics/samples.jsonl
```

## Running the pipeline

Open `notebooks/pipeline.ipynb` in Colab and run the cells top-to-bottom. Each
step is independent — you can stop after any cell and resume later because all
intermediate artifacts are on Drive.

Command-line equivalents:

```bash
python -m pipeline.step1_download --sample-size 2000000
python -m pipeline.step2_preprocess
python -m pipeline.step3_train --num-steps 12000 --batch-size 16
```
