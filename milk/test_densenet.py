from densenet import DenseNet
model = DenseNet((28, 28, 1), 
              depth_of_model=32, 
              growth_rate=32, 
              num_of_blocks=4, 
              num_layers_in_each_block=8, 
              data_format='channels_last', 
              dropout_rate=0.3, 
              pool_initial=True, 
              include_top=True,
              with_classifier=True,
              num_classes=10)
model.summary()