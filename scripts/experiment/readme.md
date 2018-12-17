# Main experiment

The set up is to test various configurations of MIL functions and encoders:

- `train_tpu_inmemory.py` runs the training, using command line flags to control the type of experiment.
    - `--mil` : one of `attention`, `average`, `instance`

The experiments are:

| # | MIL type | Encoder |
|---|-----------|---------|
| 1 | attention | pretrained |
| 2 | average | pretrained |
| 3 | instance | pretrained |
| 4 | attention | de novo |
| 5 | attention | frozen |

We'll keep track of it all with a directory structure, where individual runs are tracked by timestamp.
<!-- ```
experiment/
|____ 1_attention_pretrained/
    |____ args # record of the arguments passed to the training script
    |____ save # saved keras models 
    |____ test_lists # record of held out cases to be used for testing 
    |____ val_lists # record of cases used for 'validation'
|____ 2_average_pretrained/
|____ 3_instance_pretrained/
|____ 4_attention_denovo/
|____ 5_attention_frozen/
``` -->