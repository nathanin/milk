#!/bin/bash

# Run all of the experiments, one at a time.

set -e
steps=500
epochs=50
bag=100
lr=0.00001
pretrained=../pretraining/pretrained_reference.h5

# With attention
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained_model $pretrained
done

# Without attention (average)
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil average \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained_model $pretrained
done

# Instance classifier --> average predictions
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil instance \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained_model $pretrained
done

# Without pretraining; with attention
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs 100 \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate $lr \
        --dont_use_pretrained
done

# Freeze encoder; with average
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil average \
        --learning_rate $lr \
        --freeze_encoder \
        --deep_classifier \
        --pretrained_model $pretrained
done

# Freeze encoder; with attention
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --learning_rate $lr \
        --freeze_encoder \
        --deep_classifier \
        --pretrained_model $pretrained
done

# Use a deep classifier with trainable encoder
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --learning_rate $lr \
        --deep_classifier \
        --pretrained_model $pretrained
done