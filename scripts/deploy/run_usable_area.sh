#!/bin/bash

set -e

svs_list="tcga_prad_slides.txt"

outdir="tcga-prad-fg"
model_path="../usable_area/snapshots/mobilenet_v2"
python ./deploy_usable_area.py --model_path $model_path --slide_list $svs_list --out $outdir
