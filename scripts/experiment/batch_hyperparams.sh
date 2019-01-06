#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=500
epochs=50
bag=25

# With attention
for i in `seq 3 7`; do
    python train_tpu_inmemory.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate 1e-$i \
        --dont_use_pretrained
done