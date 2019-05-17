#!/bin/bash

set -e
steps=500
epochs=100
bag=100
lr=0.0001
pretrained=../gleason_grade/wide_model/gleason_classifier_eager.h5
batch=4
encoder=wide

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
  --encoder $encoder
