#!/bin/bash

# Run all of the experiments, one at a time.

set -e

# python train_tpu_inmemory.py \
# --steps_per_epoch 500 \
# --epochs 25 \
# --bag_size 100 \
# --mil attention \
# --pretrained_model ../pretraining/pretrained_50k.h5

# With attention
for i in `seq 1 3`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch 500 \
        --epochs 50 \
        --bag_size 150 \
        --mil attention \
        --learning_rate 0.00001 \
        --pretrained_model ../pretraining/pretrained_50k.h5
done

# Without attention (average)
for i in `seq 1 3`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch 500 \
        --epochs 50 \
        --bag_size 150 \
        --mil average \
        --learning_rate 0.00001 \
        --pretrained_model ../pretraining/pretrained_50k.h5
done

# Instance classifier --> average predictions
for i in `seq 1 3`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch 500 \
        --epochs 50 \
        --bag_size 150 \
        --mil instance \
        --learning_rate 0.00001 \
        --pretrained_model ../pretraining/pretrained_50k.h5
done

# Without pretraining; with attention
for i in `seq 1 3`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch 500 \
        --epochs 100 \
        --bag_size 150 \
        --mil attention \
        --learning_rate 0.00001 \
        --dont_use_pretrained
done

# Freeze encoder; with attention
for i in `seq 1 3`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch 500 \
        --epochs 100 \
        --bag_size 150 \
        --mil attention \
        --learning_rate 0.00001 \
        --freeze_encoder \
        --pretrained_model ../pretraining/pretrained_50k.h5
done