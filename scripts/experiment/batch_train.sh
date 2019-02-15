#!/bin/bash

# Run all of the experiments, one at a time.

set -e
steps=500
epochs=50
bag=50
lr=0.0001
pretrained=../pretraining/gleason_classifier_deep.h5

# # With attention
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch $steps \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil attention \
#         --deep_classifier \
#         --learning_rate $lr \
#         --pretrained_model $pretrained \
# 	  	  --early_stop
# done

# # Without attention (average)
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch $steps \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil average \
#         --deep_classifier \
#         --learning_rate $lr \
#         --pretrained_model $pretrained \
# 	      --early_stop
# done

# # Instance classifier --> average predictions
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch $steps \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil instance \
#         --deep_classifier \
#         --learning_rate $lr \
#         --pretrained_model $pretrained \
# 	--early_stop
# done

# Without pretraining; with attention
for i in `seq 1 1`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size 1 \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --deep_classifier \
        --learning_rate 0.0001 \
        --dont_use_pretrained \
        --early_stop \
        --seed $i \
        --accumulate 10
done

# # Freeze encoder; with average
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch $steps \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil average \
#         --learning_rate $lr \
#         --freeze_encoder \
#         --deep_classifier \
#         --pretrained_model $pretrained \
# 	      --early_stop
# done

# # Freeze encoder; with attention
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch $steps \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil attention \
#         --learning_rate $lr \
#         --freeze_encoder \
#         --deep_classifier \
#         --pretrained_model $pretrained \
# 			  --early_stop
# done

# # Enforce ensemble by setting the random seed
# for i in `seq 1 5`; do
#     python train_tpu_inmemory.py \
#         --steps_per_epoch $steps \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil attention \
#         --learning_rate $lr \
#         --deep_classifier \
#         --pretrained_model $pretrained \
#         --seed 999 \
#         --early_stop
# done
