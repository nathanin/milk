#!/bin/bash

# Outputs the all filenames in args/ along with the requested args

for f in $( ls args/*.txt ); do
    echo
    echo $f
    for a in "$@"; do
        grep $a $f
    done
done
