from __future__ import print_function
from milk.densenet import DenseNet

DENSENET_ARGS = {
## TODO
}
def make_encoder(encoder_args):
    args = {
        'depth_of_model': 32,
        'growth_rate': 32,
        'num_of_blocks': 4,
        'output_classes': 2,
        'num_layers_in_each_block': 8,
    }
    args.update(encoder_args)

    depth_of_model = args['depth_of_model']
    growth_rate = args['growth_rate']
    num_of_blocks = args['num_of_blocks']
    output_classes = args['output_classes']
    num_layers_in_each_block = args['num_layers_in_each_block']

    encoder = DenseNet(
        depth_of_model=depth_of_model,
        growth_rate=growth_rate,
        num_of_blocks=num_of_blocks,
        output_classes=output_classes,
        num_layers_in_each_block=num_layers_in_each_block,
        data_format='channels_last',
        dropout_rate=0.3,
        pool_initial=True,
        include_top=True
    )

    return encoder