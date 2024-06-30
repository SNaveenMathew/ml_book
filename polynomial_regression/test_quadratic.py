from quadratic_model import x1, y, get_model, train, x1_extended, y_extended
import numpy as np, pandas as pd, statsmodels.formula.api as smf, matplotlib.pyplot as plt, keras.backend as K, pickle
from plot_predictions import plot_pred_matrix
from model_utils import get_formula_rhs, get_summary_df, print_and_subset_summary

model = get_model()
print(model.get_weights())
model.set_weights([\
	np.array([[0.18998857, 0.39265335]], dtype=np.float32), np.array([ 0.      , -1.316958], dtype=np.float32), np.array([[-559.892 ],\
        [ 434.2152]], dtype=np.float32), np.array([185.34625], dtype=np.float32)
])
# loss ~ 0.0445
# [array([[0.18998857, 0.39265335]], dtype=float32), array([ 0.      , -1.316958], dtype=float32), array([[-559.892 ],
# [ 434.2152]], dtype=float32), array([189.08543], dtype=float32)]
# If you are not satisfied with this solution:
# model = train(model, epochs = 5000000, save_image_interval = 50000, print_epoch_interval = 50000, use_gpu = True) # Takes a long time to train
# pickle.dump(model, open("constrained_model.pkl", "wb"))
pred_matrix = pickle.load(open("bias_constrained_pred_matrix.pkl", "rb"))

# Plotting the evolution of y_pred_extended over epochs
plot_pred_matrix(pred_matrix, x1, y, x1_extended, y_extended)

# f1 = sigmoid(-1.0866661*x1 - 3.102202)
# f2 = sigmoid(0.85507196*x1 - 2.7769487)

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]
layer_outs = [func([x1_extended]) for func in functors]

f1 = layer_outs[0][0][:, 0]
plt.scatter(x1_extended, f1)
plt.show()

f2 = layer_outs[0][0][:, 1]
plt.scatter(x1_extended, f2)
plt.show()

residuals = y_extended-model.predict(x1_extended).reshape(y_extended.shape)
df = pd.DataFrame({"x1": x1_extended, "f2": f2, "f1": f1, "y": y_extended, "residuals": residuals})
f1_rmse = [0] * 10
f2_rmse = [0] * 10
f1_rmse[0] = ((df['f1'] - df['f1'].mean())**2).mean()
f2_rmse[0] = ((df['f2'] - df['f2'].mean())**2).mean()
i_s = [i for i in range(1, 10)]
for i in i_s:
	if i > 1:
		df['x1_' + str(i)] = df['x1']**i
	formula_rhs = get_formula_rhs(df.columns)
	print(i)
	print(formula_rhs)
	f1_formula = "f1 ~ " + formula_rhs
	f2_formula = "f2 ~ " + formula_rhs
	f1_df, f1_rmse_i = get_summary_df(f1_formula, df)
	# print(i)
	# print(f1_df)
	f2_df, f2_rmse_i = get_summary_df(f2_formula, df)
	# print(f2_df)
	f1_rmse[i-1] = f1_rmse_i
	f2_rmse[i-1] = f2_rmse_i

i_s = [0] + i_s
plt.scatter(i_s, f1_rmse)
plt.show()

plt.scatter(i_s, f2_rmse)
plt.show()

# Choose a suitable elbow curve and put values here
f1_pow = 2
f2_pow = 3
for i in range(2, max(f1_pow, f2_pow) + 1):
	df["x1_" + str(i)] = df["x1"]**i

f1_formula = " + ".join(["x1_" + str(i) for i in range(2, f1_pow + 1)])
f1_formula = 'f1 ~ x1 + ' + f1_formula if f1_formula != '' else 'f1 ~ x1'
f1_model = smf.ols(formula = f1_formula, data = df).fit()
f1_model_df = get_summary_df(f1_formula, df)
f1_model_df = print_and_subset_summary(f1_model_df, set_variable_index = True)
f1_model.summary()
# f1 = 0.5000 + 0.0235*x

f2_formula = " + ".join(["x1_" + str(i) for i in range(2, f2_pow + 1)])
f2_formula = 'f2 ~ x1 + ' + f2_formula if f2_formula != '' else 'f2 ~ x1'
f2_model = smf.ols(formula = f2_formula, data = df).fit()
f2_model_df = get_summary_df(f2_formula, df)
f2_model_df = print_and_subset_summary(f2_model_df, set_variable_index = True)
f2_model.summary()
# f2 =  0.02116 + 0.0327*x + 0.0017*x^2

print(np.corrcoef(df['f1'], df['f2'])[1, 0]) # not as correlated as the previous model
# 0.9890336214505975
print(np.corrcoef(df['residuals'], df['f1'])[1, 0])
# -0.028142472036833815
print(np.corrcoef(df['residuals'], df['f2'])[1, 0])
# 0.09091332427367725

# Final model approximation of y:
final_layer_model = smf.ols(formula = 'y ~ f1 + f2', data = df).fit()
final_layer_model_df = get_summary_df('y ~ f1 + f2', df)
final_layer_model_df = print_and_subset_summary(final_layer_model_df, set_variable_index = True)
final_layer_model.summary()
# y = 797.5177 - 2329.4654*f1 + 1740.3984*f2
print(np.corrcoef(final_layer_model.resid, df['f1']))
# [[ 1.0000000e+00 -4.0642508e-15]
#  [-4.0642508e-15  1.0000000e+00]]
print(np.corrcoef(final_layer_model.resid, df['f2']))
# [[ 1.00000000e+00 -4.34293746e-15]
#  [-4.34293746e-15  1.00000000e+00]]

model_wts = model.get_weights()
use_lm_wts = True

if use_lm_wts:
	final_y_hat_coefs = final_layer_model_df['coef']['f1']*f1_model_df['coef']
	for var in f2_model_df.index:
		try:
			final_y_hat_coefs[var] += final_layer_model_df['coef']['f2']*f2_model_df['coef'][var]
		except:
			final_y_hat_coefs[var] = final_layer_model_df['coef']['f2']*f2_model_df['coef'][var]

	final_y_hat_coefs['Intercept'] += final_layer_model_df['coef']['Intercept']
else:
	final_y_hat_coefs = model_wts[2][0][0]*f1_model_df['coef']
	for var in f2_model_df.index:
		try:
			final_y_hat_coefs[var] += model_wts[2][1][0]*f2_model_df['coef'][var]
		except:
			final_y_hat_coefs[var] = model_wts[2][1][0]*f2_model_df['coef'][var]

	final_y_hat_coefs['Intercept'] += model_wts[3][0]

print(final_y_hat_coefs)
# With final_layer_model
# Variable
# Intercept    1.053301
# x1           2.168591
# x1_2         2.958677

# With tensorflow-keras trained model weights
# Variable
# Intercept   -2.719824
# x1           1.041375
# x1_2         0.738166

# Keeping only the quadratic term:
f1_quadratic_formula = 'f1 ~ x1 + x1_2'
f1_quadratic_model_df = get_summary_df(f1_quadratic_formula, df)
f1_quadratic_model_df = print_and_subset_summary(f1_quadratic_model_df)
# f1 = 0.500000 + 0.023500*x1 - 0.000006*x1_2

f2_quadratic_formula = 'f2 ~ x1 + x1_2'
f2_quadratic_model_df = get_summary_df(f2_quadratic_formula, df)
f2_quadratic_model_df = print_and_subset_summary(f2_quadratic_model_df)
# f2 = 0.2116 + 0.0327*x1 - 0.0017*x1_2

final_layer_model.summary()
# y = 797.5177 - 2329.4654*f1 + 1740.3984*f2
# Final model approximation of y:

final_y_hat_coefs = final_layer_model_df['coef']['f1']*f1_quadratic_model_df['coef'] + final_layer_model_df['coef']['f2']*f2_quadratic_model_df['coef']
final_y_hat_coefs['Intercept'] += final_layer_model_df['coef']['Intercept']
print(final_y_hat_coefs)
# Variable
# Intercept    1.053301
# x1           2.168591
# x1_2         2.971853
# Name: coef, dtype: float64

df['y_pred'] = df['y'] - df['residuals']
df['data_type'] = 'in_range'
df['data_type'][(df['x1'] < x1.min()) | (df['x1'] > x1.max())] = 'out_of_range'
df['MSE'] = df['residuals']**2
print(df['MSE'].mean())
print(df.groupby(['data_type'])['MSE'].mean())
