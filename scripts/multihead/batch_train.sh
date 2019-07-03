#!/bin/bash

set -e
steps=2000
epochs=100
bag=200
lr=0.0001
pretrained=../../gleason_grade/shallow_model/gleason_classifier_eager.5x.shallow.h5
heads=5
batch=1
encoder=shallow
data=../../dataset/tiles_10x_epithelium
datal=../../dataset/tiles_10x_epithelium/CASE_LABEL_DICT.pkl

python train_eager.py \
  --steps_per_epoch $steps \
  --batch_size $batch \
  --epochs $epochs \
  --bag_size $bag \
  --mil attention \
  --deep_classifier \
  --heads $heads \
  --learning_rate $lr \
  --early_stop \
  --seed 1 \
  --temperature 0.5 \
  --pretrained $pretrained \
  --encoder $encoder \
  --data_patt $data \
  --data_labels $datal
