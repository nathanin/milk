#!/bin/bash

set -e

# module_name=mobilenet_v2
# module_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/classification/2"

module_name=mobilenet_v2
module_url="https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/2"

# module_name=resnet_50
# module_url="https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1"

python retrain.py --image_dir ./training_tiles \
--summaries_dir ./logs_10X/$module_name \
--bottleneck_dir ./bottlenecks_10X/$module_name \
--tfhub_module $module_url \
--saved_model_dir ./snapshots_10X/$module_name \
--how_many_training_steps 5500 \
--learning_rate 0.0001 \
--train_batch_size 16
# --flip_left_right 1 \
# --random_brightness 10 \
