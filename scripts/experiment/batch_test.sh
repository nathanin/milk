#!/bin/bash

set -e

# testtype=npy
# testdir=test_lists
# python batch_test.py --test $testtype --odir result_test --testdir $testdir --run_list runs_mil.txt

### path including the experiment directory
runlist=(
  runs_mil.txt
  runs_average.txt
  runs_instance.txt
  runs_frozen_average.txt
  runs_frozen_attention.txt
  runs_ensemble.txt
)
odir=result_test
testdir=test_lists
parallel --dry-run -j 3 "python batch_test.py --test $testtype --odir $odir --run_list {}" ::: ${runlist[@]}
parallel -j 3 "python batch_test.py --test $testtype --odir $odir --testdir $testdir --run_list {}" ::: ${runlist[@]}

odir=result_val
testdir=val_lists
parallel -j 3 "python batch_test.py --test $testtype --odir $odir --testdir $testdir --run_list {}" ::: ${runlist[@]}

odir=result_test_mcdrop
testdir=test_lists
parallel -j 3 "python batch_test.py --test $testtype --mcdropout --odir $odir --testdir $testdir --run_list {}" ::: ${runlist[@]}

odir=result_val_mcdrop
testdir=val_lists
parallel -j 3 "python batch_test.py --test $testtype --mcdropout --odir $odir --testdir $testdir --run_list {}" ::: ${runlist[@]}

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
