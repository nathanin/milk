#!/bin/bash

echo $1
for pth in $( ls args/*.txt ); do
  p=${pth##*/}
  p=${p%.*}

  if grep -xq mil.$1 $pth
  then
    echo testing $p
    python test_eager.py --timestamp $p --odir result_test_mcdrop --mil $1 --mcdropout
  else
    echo skipping $p
  fi
done

