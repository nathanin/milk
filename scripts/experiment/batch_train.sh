#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=1000
epochs=100
bag=25
lr=0.0001
pretrained=../gleason_grade/gleason_classifier_shallow/gleason_classifier.h5
batch=4
encoder=shallow

# with pretraining; with attention
for i in `seq 1 3`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained $pretrained \
        --early_stop \
        --seed $i \
        --accumulate 5 \
        --temperature 0.5 \
        --encoder $encoder
done
