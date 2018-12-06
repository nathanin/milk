from __future__ import print_function
from densenet import DenseNet

def make_encoder(image, input_shape, encoder_args=None):
    args = {
        'depth_of_model': 32,
        'growth_rate': 32,
        'num_of_blocks': 4,
        'num_layers_in_each_block': 8,
        'dropout_rate': 0.3
    }
    if encoder_args is not None:
        args.update(encoder_args)

    depth_of_model = args['depth_of_model']
    growth_rate = args['growth_rate']
    num_of_blocks = args['num_of_blocks']
    num_layers_in_each_block = args['num_layers_in_each_block']
    dropout_rate = args['dropout_rate']

    encoder = DenseNet(
        image = image,
        input_shape=input_shape,
        depth_of_model=depth_of_model,
        growth_rate=growth_rate,
        num_of_blocks=num_of_blocks,
        num_layers_in_each_block=num_layers_in_each_block,
        data_format='channels_last',
        dropout_rate=dropout_rate,
        pool_initial=True,
        include_top=True
    )

    return encoder