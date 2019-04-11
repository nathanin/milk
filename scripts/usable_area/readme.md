## Procedure for identifying the "usable area" in a set of slides

This reduces to fitting a binary classifier to sort single fields of view into "un-usable" and "usable" sets.
We take advantage of [TensorFlow HUB](https://tfhub.dev) models and specifications with the assumption that the distinction between un-usable/usable areas will readily be learned by a linear classifier over some pretrained features.
Thanks to the high level abstraction priveded by TFHUB, the total time from slide --> model --> inference is on the order of hours, down to 10's of minutes in extreme cases.

## Data
We collect large chunks of data, as large as possible, containing both common artifacts, and typified usable areas.
Special care should be taken to avoid corner-artifacts, as tissues are typically not mounted orthogonal to the x-y plane.
Minimizing these artifacts now simplifies downstream data curation.
The source images are stored in some directory with a structure like:

```
training_bigimg/
|_____ 0/ # <---- un-usable
|_____ 1/ # <---- usable
```

We ammed the script `create_tfhub_training.py` to split the big images into individual tiles sized according to a TFHUB model input specification.
The output is written to a new `training_tiles` directory with the same structure, except now we've split out many small tiles from our original large regions.

## Retraining a TFHUB model with retrain.py wrapper
Execute `run_retrain.sh`, having specified targets to save the TFHUB-generated bottlenecks, training logs, and model snapshots.

In the default case, the majority of compute will be devoted to running the specified model forward to generate bottlenecks, followed by a rapid and satisfying training procedure for our appended linear classifier.

## Deploying the model with `run_deploy.sh`
Using the included wrapper for `deploy_retrained.py`, we can run inference on a set of Aperio SVS format images.
We require `svs_reader` to interface with the Aperio images. 

```
$ cd <somewhere>
$ source activate my_env
(my_env) $ git clone https://github.com/BioImageInformatics/svs_reader
(my_env) $ cd svs_reader
(my_env) $ pip install -e .
```

Now, call `run_deploy.sh` with a single argument that is a path to some local directory housing a set of Aperio images.
By default we write predicted "usable area" images to a directory called `inference`. 

Inspect the result, and edit the training set, or image magnifications as needed.
