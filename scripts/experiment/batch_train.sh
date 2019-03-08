#!/bin/bash

set -e
steps=1000
epochs=100
bag=50
lr=0.001
pretrained=../gleason_grade/shallow_model/gleason_classifier.h5
batch=3
encoder=shallow

# with pretraining; with attention
for i in `seq 1 3`; do
    python train_keras.py \
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
        --encoder $encoder
done

for i in `seq 1 3`; do
    python train_keras.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil average \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained $pretrained \
        --early_stop \
        --seed $i \
        --temperature 0.5 \
        --accumulate 5 \
        --encoder $encoder
done

for i in `seq 1 3`; do
    python train_keras.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil instance \
        --deep_classifier \
        --learning_rate $lr \
        --early_stop \
        --seed $i \
        --temperature 0.5 \
        --accumulate 5 \
        --encoder $encoder
done
