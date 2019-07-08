#!/bin/bash

set -e

epochs=100
lr=0.0001
pretrained=save/2019_07_05_19_58_56.h5
heads=5
batch=1
encoder=small
crop_size=128
dataset=../../dataset2/pnbx-10x-chunk.h5

bags=( 1 10 25 50 75 100 150 200 )
for b in ${bags[@]}; do 
  bag_size=${b}

  out=${pretrained}.bag${b}.1.csv
  time python ../test_eager.py \
    --dataset $dataset \
    --out ${out} \
    --batch_size $batch \
    --epochs $epochs \
    --bag_size $bag_size \
    --crop_size $crop_size \
    --mil attention \
    --heads $heads \
    --learning_rate $lr \
    --early_stop \
    --seed 999 \
    --temperature 0.5 \
    --encoder $encoder \
    --pretrained_model $pretrained
done