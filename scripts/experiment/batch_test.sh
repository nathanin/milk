#!/bin/bash

set -e

# for td in $( ls save ); do
#     timestamp=${td%.*}
#     echo $timestamp
#     python ./test_npy.py \
#         --timestamp $timestamp \
#         --odir result \
#         --testdir val_lists
# done

for td in $( ls save ); do
    timestamp=${td%.*}
    echo $timestamp
    python ./test_svs.py \
        --timestamp $timestamp \
        --odir result_mcdropout \
        --mcdropout \
        --testdir test_lists
done

for td in $( ls save ); do
    timestamp=${td%.*}
    echo $timestamp
    python ./test_svs.py \
        --timestamp $timestamp \
        --odir result \
        --testdir test_lists
done