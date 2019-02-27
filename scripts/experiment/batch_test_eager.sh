#!/bin/bash

for pth in $( ls args/*.txt ); do
  p=${pth##*/}
  p=${p%.*}
  echo $p

  if grep -xq mil.$1 $pth
  then
    python test_eager.py --timestamp $p --odir result_test_mcdrop --mil $1 --mcdropout
  else
    echo skipping file
  fi


done
