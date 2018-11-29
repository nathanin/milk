#!/bin/bash

set -e

for td in $( ls save ); do
    echo $td
    python ./test_npy.py \
        --snapshot_dir save/$td \
        --test_list test_lists/${td}.txt \
        --n_repeat 1 \
        --savepath figures/auc_mcdrop_${td}.png \
        --mcdropout 10
done