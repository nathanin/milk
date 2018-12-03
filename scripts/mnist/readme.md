## Bagged MNIST Example

This is a minimum working example for the "bagged MNIST" problem.

#### Problem statement:

Instead of classifying individual digits, we want to find out if a **set** of digits contains a example of a **particular** digit. 
We decide which digit(s) is/are considered positive, but we don't label individual digits as positive or negative, we label the whole **set** as positive or negative. 

For instance, pick the digit `9` to be the positive instance. 
We draw two sets of mnist digits. Set 1: `(0, 1, 2, 3, 4)` , and Set 2: `(5, 6, 7, 8, 9)`. 
Set number 2 is a positive set (labelled `1`), and Set number 1 is a negative set (labelled `0`).
At training time we give the digits in a batched form with shape `[batch, N, h, w, c]`, and a set of labels with shape `[batch, 2]`.


#### Solution:

Run the script `train_mnist.py` and see how long it takes to converge.

To improve convergence, we might pretrain the `encoder` with a target we think is going to be closely related to the multiple instance target. 
Straightforwardly, the encoder is trained as an mnist digit classifier.

Run the script `pretrain_mnist.py`, which populates a directory with some snapshots.
Then, rerun the MIL training, this time using the pretrained model as a head start: `train_mnist.py --initial_weights ./trained/classifier-xxxx`, replacing `xxxx` according to how long the classifier was trained. 

Hopefully the loss converges, and testing accuracy becomes acceptable, much more quickly. 