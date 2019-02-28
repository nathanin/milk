#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=1000
epochs=100
bag=50
lr=0.0001
pretrained=../gleason_grade/gleason_grade_big_model.h5
batch=1
encoder=shallow

# Without pretraining; with attention
for i in `seq 1 3`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate 0.0001 \
        --dont_use_pretrained \
        --early_stop \
        --seed $i \
        --accumulate 10 \
        --temperature 0.1 \
        --encoder $encoder
done