#!/bin/bash

set -e

testtype=npy

runlist=(
  runs_mil.txt,
  runs_average.txt,
  runs_instance.txt,
  runs_frozen_average.txt,
  runs_frozen_attention.txt,
  runs_ensemble.txt
)

parallel -j 3 "python batch_test.py --test npy --odir result_test --run_list {}" ::: ${runlist[@]}
parallel -j 3 "python batch_test.py --test npy --val --odir result_val --run_list {}" ::: ${runlist[@]}
parallel -j 3 "python batch_test.py --test npy --mcdropout --odir result_test_mcdrop --run_list {}" ::: ${runlist[@]}
parallel -j 3 "python batch_test.py --test npy --mcdropout --val --odir result_val_mcdrop --run_list {}" ::: ${runlist[@]}

# ## "test"
# python batch_test.py --test $testtype --odir result_test --run_list runs_mil.txt
# python batch_test.py --test $testtype --odir result_test --run_list runs_average.txt
# python batch_test.py --test $testtype --odir result_test --run_list runs_instance.txt
# #python batch_test.py --test $testtype --odir result_test --run_list runs_end2end.txt
# python batch_test.py --test $testtype --odir result_test --run_list runs_frozen_average.txt
# python batch_test.py --test $testtype --odir result_test --run_list runs_frozen_attention.txt
# python batch_test.py --test $testtype --odir result_test --run_list runs_ensemble.txt
# #python batch_test.py --test $testtype --odir result_test --run_list runs_attention_ensemble.txt

# ## "val"
# python batch_test.py --test $testtype --odir result_val --val --run_list runs_mil.txt
# python batch_test.py --test $testtype --odir result_val --val --run_list runs_average.txt
# python batch_test.py --test $testtype --odir result_val --val --run_list runs_instance.txt
# #python batch_test.py --test $testtype --odir result_val --val --run_list runs_end2end.txt
# python batch_test.py --test $testtype --odir result_val --val --run_list runs_frozen_average.txt
# python batch_test.py --test $testtype --odir result_val --val --run_list runs_frozen_attention.txt
# #python batch_test.py --test $testtype --odir result_val --val --run_list runs_ensemble.txt

# ## "test"
# python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_mil.txt
# python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_average.txt
# python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_instance.txt
# #python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_end2end.txt
# python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_frozen_average.txt
# python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_frozen_attention.txt
# #python batch_test.py --test $testtype --odir result_test_mcdrop --mcdropout --run_list runs_ensemble.txt

# ## "val"
# python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_mil.txt
# python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_average.txt
# python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_instance.txt
# #python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_end2end.txt
# python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_frozen_average.txt
# python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_frozen_attention.txt
# #python batch_test.py --test $testtype --odir result_val_mcdrop --mcdropout --val --run_list runs_ensemble.txt
