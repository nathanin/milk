wide_args = {
    'depth_of_model': 48,
    'growth_rate': 64,
    'num_of_blocks': 4,
    'num_layers_in_each_block': 12,
}

deep_args = {
    'depth_of_model': 64,
    'growth_rate': 24,
    'num_of_blocks': 4,
    'num_layers_in_each_block': 16,
}

small_args = {
    'depth_of_model': 24,
    'growth_rate': 24,
    'num_of_blocks': 4,
    'num_layers_in_each_block': 6,
}

tiny_args = {
    'depth_of_model': 12,
    'growth_rate': 36,
    'num_of_blocks': 4,
    'num_layers_in_each_block': 3,
}

big_args = {
    'depth_of_model': 80,
    'growth_rate': 36,
    'num_of_blocks': 4,
    'num_layers_in_each_block': 20,
}

shallow_args = {
    'depth_of_model': 32,
    'growth_rate': 48,
    'num_of_blocks': 4,
    'num_layers_in_each_block': 8,
}

mnist_args = {
    'depth_of_model': 16,
    'growth_rate': 24,
    'num_of_blocks': 2,
    'num_layers_in_each_block': 8,
    'pool_initial': True
}

def get_encoder_args(arg_str):
  if arg_str == 'big':
    return big_args
  elif arg_str == 'small':
    return small_args
  elif arg_str == 'tiny':
    return tiny_args
  elif arg_str == 'wide':
    return wide_args
  elif arg_str == 'deep':
    return deep_args
  elif arg_str == 'shallow':
    return shallow_args
  elif arg_str == 'mnist':
    return mnist_args
