#!/bin/bash

set -e
out_base=debug_graph
epochs=100
bag=100
lr=0.0001
heads=10
batch=1
encoder=shallow
crop_size=128
dataset=../dataset2/pnbx-10x-nochunk.h5

python ./train_eager.py \
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
  --encoder $encoder \
  --out_base $out_base
