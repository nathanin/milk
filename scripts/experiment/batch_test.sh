#!/bin/bash

set -e

# for td in $( ls save ); do
#     timestamp=${td%.*}
#     echo $timestamp
#     python ./test_npy.py \
#         --timestamp $timestamp \
#         --mcdropout \
#         --savepath val_result_mcdropout \
#         --testdir val_lists
# done

for td in $( ls save ); do
    timestamp=${td%.*}
    echo $timestamp
    python ./test_svs.py \
        --timestamp $timestamp \
        --odir processed_slides \
        --testdir test_lists
done