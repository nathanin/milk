#!/bin/bash

set -e

for i in `seq 1 5`; 
do
    python train_v1.py
done