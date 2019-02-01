#!/bin/bash
set -e

# python attention_maps.py --timestamp 2019_01_23_22_01_59 --deep_classifier
# python attention_maps.py --timestamp 2019_01_23_22_21_15 --deep_classifier
# python attention_maps.py --timestamp 2019_01_23_22_40_32 --deep_classifier
# python attention_maps.py --timestamp 2019_01_23_22_59_47 --deep_classifier
# python attention_maps.py --timestamp 2019_01_23_23_10_53 --deep_classifier

timestamps=(
  2019_01_23_22_01_59
  2019_01_23_22_21_15
  2019_01_23_22_40_32
  2019_01_23_22_59_47
  2019_01_23_23_10_53
)

for ts in ${timestamps[@]}; do
  echo $ts
done

# parallel --dry-run "python attention_maps.py --timestamp {} --deep_classifier" ::: ${timestamps[@]}
parallel --bar -j 3 "python attention_maps.py --timestamp {} --deep_classifier --randomize --batch_size 16" ::: ${timestamps[@]}
