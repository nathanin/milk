#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=1000
epochs=100
bag=50
lr=0.0001
pretrained=../pretraining/gleason_grade_big_model.h5
batch=1

## With attention
#for i in `seq 1 3`; do
#    #python train_tpu_inmemory.py \
#    python train_eager.py \
#        --batch_size $batch \
#        --steps_per_epoch $steps \
#        --epochs $epochs \
#        --bag_size $bag \
#        --mil attention \
#        --deep_classifier \
#        --learning_rate $lr \
#        --pretrained_model $pretrained \
#	  	  --early_stop \
#        --seed $i \
#        --accumulate 10
#done

# Without attention (average)
for i in `seq 1 3`; do
    # python train_tpu_inmemory.py \
    python train_eager.py \
        --batch_size $batch \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil average \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained_model $pretrained \
	      --early_stop \
        --seed $i \
        --accumulate 10
done

# Instance classifier --> average predictions
for i in `seq 1 3`; do
    # python train_tpu_inmemory.py \
    python train_eager.py \
        --batch_size $batch \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil instance \
        --deep_classifier \
        --learning_rate $lr \
        --pretrained_model $pretrained \
	      --early_stop \
        --seed $i \
        --accumulate 10
done

# # Without pretraining; with attention
# for i in `seq 1 3`; do
#     python train_eager.py \
#         --steps_per_epoch $steps \
#         --batch_size $batch \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil attention \
#         --deep_classifier \
#         --learning_rate 0.0001 \
#         --dont_use_pretrained \
#         --early_stop \
#         --seed $i \
#         --accumulate 10
# done
# 
# # Without pretraining; with attention
# for i in `seq 1 3`; do
#     python train_eager.py \
#         --steps_per_epoch $steps \
#         --batch_size $batch \
#         --epochs $epochs \
#         --bag_size $bag \
#         --mil average \
#         --deep_classifier \
#         --learning_rate 0.0001 \
#         --dont_use_pretrained \
#         --early_stop \
#         --seed $i \
#         --accumulate 10
# done

# Freeze encoder; with average
for i in `seq 1 3`; do
    # python train_tpu_inmemory.py \
    python train_eager.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil average \
        --learning_rate $lr \
        --freeze_encoder \
        --deep_classifier \
        --pretrained_model $pretrained \
	      --early_stop \
        --seed $i \
        --accumulate 10
done

# Freeze encoder; with attention
for i in `seq 1 3`; do
    # python train_tpu_inmemory.py \
    python train_eager.py \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --learning_rate $lr \
        --freeze_encoder \
        --deep_classifier \
        --pretrained_model $pretrained \
			  --early_stop \
        --seed $i \
        --accumulate 10
done

# Enforce ensemble by setting the random seed
for i in `seq 1 3`; do
    # python train_tpu_inmemory.py \
    python train_eager.py \
        --batch_size $batch \
        --steps_per_epoch $steps \
        --epochs $epochs \
        --bag_size $bag \
        --mil attention \
        --learning_rate $lr \
        --deep_classifier \
        --pretrained_model $pretrained \
        --seed 999 \
        --early_stop \
        --accumulate 10
done
