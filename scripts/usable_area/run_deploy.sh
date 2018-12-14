#!/bin/bash

set -e

# svs_dir=/media/ing/D/svs/VA
svs_dir="/media/labshare/_VA_slides_/VA_slides_40x_3rd_batch_M0only/Scanned Slides/"

outdir="inference"
# model_path="snapshots/resnet_50"
model_path="snapshots/mobilenet_v2"
python ./deploy_retrained.py --model_path $model_path --slide "$svs_dir" --out $outdir
