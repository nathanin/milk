from __future__ import print_function
from milk.densenet import DenseNet

DENSENET_ARGS = {
## TODO
}
def make_encoder(**kwargs):

    encoder = DenseNet(
        depth_of_model=20,
        growth_rate=32,
        num_of_blocks=4,
        output_classes=2,
        num_layers_in_each_block=5,
        data_format='channels_last',
        dropout_rate=0.3,
        pool_initial=True,
        include_top=True
    )

    return encoder