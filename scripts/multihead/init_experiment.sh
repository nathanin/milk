#! /bin/bash

dst=$1
echo $dst

set -e

mkdir $dst

for arg in $( cat initdirs.txt ); do
  echo $dst/$arg
  mkdir $dst/$arg
done

for arg in $( cat initfiles.txt ); do
  echo $dst/$arg
  cp $arg $dst/$arg
done

cd $dst
