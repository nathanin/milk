#!/bin/bash

set -e
steps=1000
epochs=100
bag=250
lr=0.0001
pretrained=../gleason_grade/wide_model/gleason_classifier_eager.h5
batch=1
encoder=tiny

for i in `seq 1 5`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate $lr \
        --early_stop \
        --seed $i \
        --temperature 0.5 \
        --accumulate 5 \
        --encoder $encoder \
        --data_patt ../../dataset/tiles_10x_combined
done
