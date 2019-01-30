# as long as growth-rate is the same as the main model,
# keras will be able to load by name
encoder_args ={
    'depth_of_model': 64,
    'growth_rate': 24,
    'num_of_blocks': 4,
    'output_classes': 10,
    'num_layers_in_each_block': 16,
    'pool_initial': False,
}
