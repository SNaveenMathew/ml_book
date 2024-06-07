import tensorflow.keras as keras, numpy as np, tensorflow as tf, pandas as pd, pickle
import keras.backend as K
from keras.layers import BatchNormalization, Dense
from keras import Sequential, Input
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import GlorotNormal, HeNormal
import matplotlib.pyplot as plt
tf.random.set_seed(1)
np.random.seed(1)
x1 = np.random.randn(10000, )
x1_min, x1_max = x1.min(), x1.max()
x1_extended = 2 * x1
x1_extended_in_x1_range = (x1_extended >= x1_min) & (x1_extended <= x1_max)
colors = pd.Series(['red'] * len(x1))
colors[x1_extended_in_x1_range] = 'blue'
y = 1 + 2*x1 + 3*x1**2
y_extended = 1 + 2*x1_extended + 3*x1_extended**2
biases = K.constant([0., np.log(2 - np.sqrt(3))]) # From https://www.wolframalpha.com/input/?i=maximize+d%5E2%2Fdx%5E2%28sigmoid%28x%29%2

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

def get_model(bias_constraint = True, batch_normalization = False, learning_rate = 0.01 * np.sqrt(10)):
	if batch_normalization:
		layer1 = BatchNormalization()

	if bias_constraint:
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
	else:
		layer2 = Dense(
			input_dim = 1,
			units = 2,
			use_bias = True,
			activation = "sigmoid",
			kernel_initializer = HeNormal(seed = 1)
		)

	if batch_normalization:
		layer3 = BatchNormalization()

	layer4 = Dense(
		1,
		use_bias = True,
		activation = "linear"
	)

	if batch_normalization:
		model = Sequential(
			[layer1, layer2, layer3, layer4]
		)
	else:
		model = Sequential(
			[layer2, layer4]
		)

	optimizer = Adam(learning_rate = learning_rate)
	# reduce_lr = ReduceLROnPlateau()
	model.compile(loss = 'mean_squared_error', optimizer = optimizer)
	return model

def train(model, epochs = 5000000, save_image_interval = None, print_epoch_interval = None, use_gpu = False, bias_constraint = True):
	if save_image_interval is not None:
		pred_matrix = np.ones((y.shape[0], int(np.ceil(epochs/save_image_interval))))
		idx = 0

	for epoch in range(epochs):
		# model.fit(x = x1, y = y, batch_size = 10000, callbacks=[reduce_lr])
		if use_gpu:
			with tf.device('/GPU:0'):
				model.fit(x = x1, y = y, batch_size = 10000)
		else:
			model.fit(x = x1, y = y, batch_size = 10000)

		if save_image_interval is not None:
			if epoch % save_image_interval == 0:
				y_pred_extended = model.predict(x1_extended).reshape(y.shape)
				pred_matrix[:, idx] = y_pred_extended
				idx += 1
				# tmp_df = pd.DataFrame({"x1": x1_extended, "y_pred": y_pred_extended, "y": y_extended, "color": colors})
				# tmp_df = tmp_df.sort_values(['x1']).reset_index(drop = True)
				# ax = tmp_df.plot.scatter(x='x1_extended', y='y_pred_extended', c='color', s=1)
				# fig = ax.get_figure()
				# fig = plt.figure()
				# plt.scatter(x = tmp_df['x1'], y = tmp_df['y_pred'], color = tmp_df['color'], s = 1)
				# plt.plot(tmp_df['x1'], tmp_df['y'])
				# plt.ylim(min(tmp_df['y_pred'].min(), tmp_df['y'].min()), max(tmp_df['y_pred'].max(), tmp_df['y'].max()))
				# fig.savefig("epoch_" + str(epoch + 1) + ".png")
				# plt.close(fig)

		if print_epoch_interval is not None:
			if epoch % print_epoch_interval == 0:
				print(epoch)

	if save_image_interval is not None:
		if bias_constraint:
			pickle.dump(pred_matrix, open("bias_constrained_pred_matrix.pkl", "wb"))
		else:
			pickle.dump(pred_matrix, open("bias_unconstrained_pred_matrix.pkl", "wb"))

	return model

