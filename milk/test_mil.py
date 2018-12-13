from mil import Milk

model = Milk(
    input_shape = (100, 96, 96, 3),
)

model.summary()