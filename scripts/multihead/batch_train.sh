#!/bin/bash

set -e
epochs=100
bag=50
lr=0.0001
pretrained=../../gleason_grade/shallow_model/gleason_classifier_eager.5x.shallow.h5
heads=5
batch=1
encoder=big
crop_size=128
dataset=../../dataset2/pnbx-10x-chunk.h5

python ../train_eager.py \
  --dataset $dataset \
  --batch_size $batch \
  --epochs $epochs \
  --bag_size $bag \
  --crop_size $crop_size \
  --mil attention \
  --deep_classifier \
  --heads $heads \
  --learning_rate $lr \
  --early_stop \
  --seed 999 \
  --temperature 0.5 \
  --encoder $encoder 
