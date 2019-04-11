#!/bin/bash

set -e

# svs_dir=/media/ing/D/svs/VA
svs_dir=../dataset/svs
outdir="inference_10X"
# model_path="snapshots/resnet_50"
model_path="snapshots_10X/mobilenet_v2"
python ./deploy_retrained.py --model_path $model_path --slide "$svs_dir" --out $outdir
