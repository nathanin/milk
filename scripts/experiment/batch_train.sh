#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=1000
epochs=100
bag=50
lr=0.0001
batch=1

# Without pretraining;
for i in `seq 1 3`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --learning_rate $lr \
        --deep_classifier \
        --dont_use_pretrained \
        --accumulate 10 \
        --seed $i \
        --early_stop
done

# Without pretraining;
for i in `seq 1 3`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil average \
        --learning_rate $lr \
        --deep_classifier \
        --dont_use_pretrained \
        --accumulate 10 \
        --seed $i \
        --early_stop
done

# Without pretraining;
for i in `seq 1 3`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
        --mil instance \
        --deep_classifier \
        --learning_rate $lr \
        --dont_use_pretrained \
        --accumulate 10 \
        --seed $i \
        --early_stop
done
