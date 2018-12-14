#!/bin/bash

# Run all of the experiments, one at a time.

set -e

# With attention
for i in `seq 1 5`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch 250 \
        --epochs 50 \
        --bag_size 100 \
        --mil attention \
        --pretrained_model ../pretrained/pretrained_50k.h5
done

# # Without attention (average)
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch 1000 \
#         --bag_size 100 \
#         --mil average
# done

# # Instance classifier --> average predictions
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch 1000 \
#         --bag_size 100 \
#         --mil instance
# done

# # Without pretraining; with attention
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch 1000 \
#         --bag_size 100 \
#         --mil attention \
#         --dont_use_pretrained
# done

# # Freeze encoder; with attention
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch 1000 \
#         --bag_size 100 \
#         --mil attention \
#         --freeze_encoder
# done