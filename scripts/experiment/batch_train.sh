#!/bin/bash

# Run all of the experiments, one at a time.
set -e
steps=1000
epochs=100
bag=50
lr=0.0001
<<<<<<< HEAD
=======
pretrained=../gleason_grade/gleason_grade_big_model.h5
>>>>>>> f6a4579d300924d13dcf9bac07929e779bed7395
batch=1
encoder=shallow

<<<<<<< HEAD
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
=======
# Without pretraining; with attention
>>>>>>> f6a4579d300924d13dcf9bac07929e779bed7395
for i in `seq 1 3`; do
    python train_eager.py \
        --steps_per_epoch $steps \
        --batch_size $batch \
        --epochs $epochs \
        --bag_size $bag \
<<<<<<< HEAD
        --mil instance \
        --deep_classifier \
        --learning_rate $lr \
        --dont_use_pretrained \
        --accumulate 10 \
        --seed $i \
        --early_stop
done
=======
        --mil attention \
        --deep_classifier \
        --learning_rate 0.0001 \
        --dont_use_pretrained \
        --early_stop \
        --seed $i \
        --accumulate 10 \
        --temperature 0.1 \
        --encoder $encoder
done
>>>>>>> f6a4579d300924d13dcf9bac07929e779bed7395
