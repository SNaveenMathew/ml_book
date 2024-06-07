from quadratic_model import x1, y, get_model, train, x1_extended, y_extended
import numpy as np, statsmodels.formula.api as smf, matplotlib.pyplot as plt, keras.backend as K, pandas as pd, pickle
from plot_predictions import plot_pred_matrix
from model_utils import get_formula_rhs, get_summary_df, print_and_subset_summary

model = get_model(bias_constraint = False, learning_rate = 0.1 * np.sqrt(10))
# model.get_weights()
# [array([[-1.0866661 ,  0.85507196]], dtype=float32), array([-3.102202 , -2.7769487], dtype=float32), array([[51.564045],
#        [91.12118 ]], dtype=float32), array([-6.4819527], dtype=float32)]
model.set_weights([np.array([[-1.0866661 ,  0.85507196]], dtype=np.float32), np.array([-3.102202 , -2.7769487], dtype=np.float32), np.array([[51.564045],
       [91.12118]], dtype=np.float32), np.array([-6.4819527], dtype=np.float32)])
# loss ~ 0.0118
# model = train(model, epochs = 1000000, save_image_interval = 10000, print_epoch_interval = 10000, use_gpu = True, bias_constraint = False)
pred_matrix = pickle.load(open("bias_unconstrained_pred_matrix.pkl", "rb"))

# Plotting the evolution of y_pred_extended over epochs
plot_pred_matrix(pred_matrix, x1, y, x1_extended, y_extended)

# f1 = sigmoid(-1.0866661*x1 - 3.102202)
# f2 = sigmoid(0.85507196*x1 - 2.7769487)

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp], [out]) for out in outputs]
layer_outs = [func([x1]) for func in functors]

f1 = layer_outs[0][0][:, 0]
plt.scatter(x1, f1)
plt.show()

f2 = layer_outs[0][0][:, 1]
plt.scatter(x1, f2)
plt.show()

residuals = y_extended-model.predict(x1_extended).reshape(y_extended.shape)
df = pd.DataFrame({"x1": x1_extended, "f2": f2, "f1": f1, "y": y_extended, "residuals": residuals})
f1_rmse = [0] * 9
f2_rmse = [0] * 9
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

plt.scatter(i_s, f1_rmse)
plt.show()

plt.scatter(i_s, f2_rmse)
plt.show()

# Choose a suitable elbow curve and put values here
f1_pow = 3
f2_pow = 2
for i in range(2, max(f1_pow, f2_pow) + 1):
	df["x1_" + str(i)] = df["x1"]**i

f1_formula = " + ".join(["x1_" + str(i) for i in range(2, f1_pow + 1)])
f1_formula = 'f1 ~ x1 + ' + f1_formula if f1_formula != '' else 'f1 ~ x1'
f1_model = smf.ols(formula = f1_formula, data = df).fit()
f1_model_df = get_summary_df(f1_formula, df)
f1_model_df = print_and_subset_summary(f1_model_df, set_variable_index = True)
f1_model.summary()
# f1 = 0.0390 - 0.0578*x + 0.0293*x^2 - 0.0036*x^3 - 0.0005*x^4 + 0.000076*x^5

f2_formula = " + ".join(["x1_" + str(i) for i in range(2, f2_pow + 1)])
f2_formula = 'f2 ~ x1 + ' + f2_formula if f2_formula != '' else 'f2 ~ x1'
f2_model = smf.ols(formula = f2_formula, data = df).fit()
f2_model_df = get_summary_df(f2_formula, df)
f2_model_df = print_and_subset_summary(f2_model_df, set_variable_index = True)
f2_model.summary()
# f2 =  0.0552 + 0.0629*x + 0.0218*x^2 + 0.0007*x^3 - 0.0003*x^4

print(np.corrcoef(df['f1'], df['f2'])[1, 0]) # not as correlated as the previous model
# -0.44216284304390313
print(np.corrcoef(df['residuals'], df['f1'])[1, 0])
# 0.33569181428433276
print(np.corrcoef(df['residuals'], df['f2'])[1, 0])
# 0.24449478085140286

# Final model approximation of y:
final_layer_model = smf.ols(formula = 'y ~ f1 + f2', data = df).fit()
final_layer_model_df = get_summary_df('y ~ f1 + f2', df)
final_layer_model_df = print_and_subset_summary(final_layer_model_df, set_variable_index = True)
final_layer_model.summary()
# y = -9.5662 + 66.8573*f1 + 106.9213*f2
print(np.corrcoef(final_layer_model.resid, df['f1']))
# array([[1.00000000e+00, 1.67132462e-15],
#        [1.67132462e-15, 1.00000000e+00]])
print(np.corrcoef(final_layer_model.resid, df['f2']))
# array([[ 1.00000000e+00, -5.62366943e-16],
#        [-5.62366943e-16,  1.00000000e+00]])

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
# Intercept   -1.056710
# x1           2.860998
# x1_2         4.289803
# x1_3        -0.165841
# x1_4        -0.065505
# x1_5         0.005080

# With tensorflow-keras trained model weights
# Variable
# Intercept    0.558934
# x1           2.751120
# x1_2         3.497268
# x1_3        -0.121846
# x1_4        -0.053118
# x1_5         0.003918


# Keeping only the quadratic term:
f1_quadratic_formula = 'f1 ~ x1 + x1_2'
f1_quadratic_model_df = get_summary_df(f1_quadratic_formula, df)
f1_quadratic_model_df = print_and_subset_summary(f1_quadratic_model_df)
# f1 = 0.0605 - 0.0831*x1 + 0.0182*x1_2

f2_quadratic_formula = 'f2 ~ x1 + x1_2'
f2_quadratic_model_df = get_summary_df(f2_quadratic_formula, df)
f2_quadratic_model_df = print_and_subset_summary(f2_quadratic_model_df)
# f2 = 0.0691 + 0.0710*x1 + 0.0147*x1_2

final_layer_model.summary()
# y = -9.5662 + 66.8573*f1 + 106.9213*f2
# Final model approximation of y:

final_y_hat_coefs = final_layer_model_df['coef']['f1']*f1_quadratic_model_df['coef'] + final_layer_model_df['coef']['f2']*f2_quadratic_model_df['coef']
final_y_hat_coefs['Intercept'] += final_layer_model_df['coef']['Intercept']
print(final_y_hat_coefs)
# Variable
# Intercept    1.866928
# x1           2.035571
# x1_2         2.788546
