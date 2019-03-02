#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=2500
epochs=100
bag=50
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
        --accumulate 10 \
        --temperature 0.5 \
        --encoder $encoder
done

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
        --accumulate 10 \
        --temperature 0.5 \
        --encoder $encoder \
        --freeze_encoder
done
