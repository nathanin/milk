import sys
sys.path.insert(0, '../..')
from milk import Classifier
import tensorflow as tf

tf.enable_eager_execution()

model = Classifier()
dummy_data = tf.zeros((20, 128, 128, 3), dtype=tf.float32)
dummy_y = model(dummy_data, verbose=True)
model.summary()