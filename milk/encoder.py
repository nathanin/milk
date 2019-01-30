from __future__ import print_function
from .densenet import DenseNet

def make_encoder(image, input_shape, trainable=True, encoder_args=None):
    # Default configuration
    args = {
        'depth_of_model': 32,
        'growth_rate': 32,
        'num_of_blocks': 4,
        'num_layers_in_each_block': 8,
        'dropout_rate': 0.3,
        'mcdropout': False,
        'pool_initial': True,
    }
    if encoder_args is not None:
        args.update(encoder_args)

    print('Instantiating a DenseNet with settings:')
    for k, v, in args.items():
        print('\t{:<25}: {}'.format(k, v))

    depth_of_model = args['depth_of_model']
    growth_rate = args['growth_rate']
    num_of_blocks = args['num_of_blocks']
    num_layers_in_each_block = args['num_layers_in_each_block']
    dropout_rate = args['dropout_rate']
    mcdropout = args['mcdropout']
    pool_initial = args['pool_initial']

    encoder = DenseNet(
        image          = image,
        input_shape    = input_shape,
        depth_of_model = depth_of_model,
        growth_rate    = growth_rate,
        num_of_blocks  = num_of_blocks,
        num_layers_in_each_block = num_layers_in_each_block,
        data_format    = 'channels_last',
        dropout_rate   = dropout_rate,
        pool_initial   = pool_initial,
        include_top    = True,
        mcdropout      = mcdropout,
        trainable      = trainable
    )

    return encoder
