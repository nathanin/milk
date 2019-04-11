#!/bin/bash

# Test for some condition filtered by MIL type:
# example:
#   
#   batch_test_eager.sh attention

echo $1
for pth in $( ls args/*.txt ); do
  # extract the timestamp from path
  p=${pth##*/}
  p=${p%.*}

  if grep -xq mil.$1 $pth
  then
    echo testing $p
    python test_eager.py \
      --timestamp $p \
      --odir result_test \
      --mil $1 \
      --temperature 0.5 \
      --encoder wide \
      --batch_size 32 \
      --oversample 1.5

  else
    echo skipping $p
  fi
done
