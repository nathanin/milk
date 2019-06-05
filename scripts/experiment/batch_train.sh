#!/bin/bash

set -e
steps=500
epochs=100
bag=100
lr=0.0001
pretrained=../gleason_grade/wide_model/gleason_classifier_eager.h5
batch=1
encoder=deep
data=../../dataset/tiles_10x_epithelium
datal=../../dataset/tiles_10x_epithelium/CASE_LABEL_DICT.pkl

python train_eager.py \
  --steps_per_epoch $steps \
  --batch_size $batch \
  --epochs $epochs \
  --bag_size $bag \
  --mil attention \
  --deep_classifier \
  --learning_rate $lr \
  --early_stop \
  --seed 1 \
  --temperature 0.5 \
  --pretrained $pretrained \
  --encoder $encoder \
  --data_patt $data \
  --data_labels $datal
