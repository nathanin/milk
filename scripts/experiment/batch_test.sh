#!/bin/bash

set -e

for td in $( ls no_attention/save ); do
    echo $td
    python ./test_npy.py \
        --snapshot_dir no_attention/save/$td \
        --test_list no_attention/test_lists/${td}.txt \
        --n_repeat 1 \
        --savepath no_attention/figures/auc_${td}.png 
done