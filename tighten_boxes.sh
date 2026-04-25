#!/usr/bin/env bash
set -euo pipefail

python3 train.py \
  --model runs/license_plate/quick_check_plus5/weights/best.pt \
  --imgsz 704 \
  --epochs 3 \
  --batch 8 \
  --patience 2 \
  --device auto \
  --name quick_check_plus5_tight \
  --cache ram \
  --photo-augment-ratio 0.02 \
  --optimizer AdamW \
  --lr0 0.0008 \
  --lrf 0.01 \
  --weight-decay 0.0005 \
  --box 10.0 \
  --dfl 2.0 \
  --degrees 2.0 \
  --fliplr 0.25 \
  --hsv-s 0.08 \
  --hsv-v 0.12 \
  --skip-val-predictions
