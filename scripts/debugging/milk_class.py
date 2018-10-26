import sys
sys.path.insert(0, '../..')
from milk import Milk
import tensorflow as tf

tf.enable_eager_execution()

model = Milk()
dummy_data = tf.zeros((20, 128, 128, 3), dtype=tf.float32)
dummy_y = model(dummy_data, verbose=True)
model.summary()