#!/bin/bash

for f in $( ls args/*.txt ); do
    echo
    echo $f
    for a in "$@"; do
        grep $a $f
    done
done
