#!/bin/bash
#python deploy_eager.py --odir cedars-prad \
#    --test_list cedars-prad-rp.txt \
#    --encoder wide \
#    --snapshot ../experiment/wide_model_pretrained/save/2019_03_28_19_01_12.h5 \
#    --mil attention \
#    --fgdir ../usable_area/inference \
#    --oversample 1.25

python deploy_eager.py --odir local-svs \
    --test_list local_svs.txt \
    --encoder wide \
    --snapshot ../experiment/wide_model_pretrained/save/2019_03_28_19_01_12.h5 \
    --mil attention \
    --fgdir ../usable_area/inference \
    --oversample 3

# python deploy_eager.py --odir local-svs-instance \
#     --test_list local_svs.txt \
#     --encoder wide \
#     --snapshot ../experiment/wide_model_pretrained/save/2019_03_30_01_54_30.h5 \
#     --mil instance \
#     --fgdir ../usable_area/inference \
#     --oversample 2