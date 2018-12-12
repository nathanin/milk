#!/bin/bash

set -e

for td in $( ls save ); do
    timestamp=${td%.*}
    echo $timestamp
    python ./test_npy.py \
        --timestamp $timestamp \
        --mcdropout \
        --savepath test_result_mcdropout \
        --testdir test_lists
done