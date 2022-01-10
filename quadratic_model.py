import tensorflow.keras as keras, numpy as np, tensorflow as tf
import keras.backend as K
from keras.layers import BatchNormalization, Dense
from keras import Sequential, Input
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import GlorotNormal, HeNormal
tf.random.set_seed(1)
np.random.seed(1)
x1 = np.random.randn(10000, )
y = 1 + 2*x1 + 3*x1**2
biases = K.constant([0., np.log(2 - np.sqrt(3))]) # From https://www.wolframalpha.com/input/?i=maximize+d%5E2%2Fdx%5E2%28sigmoid%28x%29%29

class ConstantTensorInitializer(keras.initializers.Initializer):
	def __init__(self, t):
		self.t = t

	def __call__(self, shape, dtype=None):
		return self.t

	def get_config(self):
		return {'t': self.t}



class ConstantTensorConstraint(keras.constraints.Constraint):
	def __init__(self, t):
		self.t = t

	def __call__(self, w):
		return self.t

	def get_config(self):
		return {'t': self.t}

def sigmoid(x):
	x = np.exp(x)
	return(x / (1 + x))

# layer1 = BatchNormalization()
layer2 = Dense(
	input_dim = 1,
	units = 2,
	use_bias = True,
	# kernel_initializer = GlorotNormal(seed = 1),
	kernel_initializer = HeNormal(seed = 1),
	bias_initializer = ConstantTensorInitializer(biases),
	bias_constraint = ConstantTensorConstraint(biases),
	activation = "sigmoid"
)
# layer2 = Dense(
# 	2,
# 	use_bias = True,
# 	activation = "sigmoid"
# )
# layer3 = BatchNormalization()
layer4 = Dense(
	1,
	use_bias = True,
	activation = "linear"
)

# model = Sequential(
# 	[layer1, layer2, layer3, layer4]
# )
model = Sequential(
	[layer2, layer4]
)

optimizer = Adam(learning_rate = 0.01 * np.sqrt(10))
# reduce_lr = ReduceLROnPlateau()
model.compile(loss = 'mean_squared_error', optimizer = optimizer)

def train(model, epochs = 5000000):
	for epoch in range(epochs):
		# model.fit(x = x1, y = y, batch_size = 10000, callbacks=[reduce_lr])
		model.fit(x = x1, y = y, batch_size = 10000)

	return model

