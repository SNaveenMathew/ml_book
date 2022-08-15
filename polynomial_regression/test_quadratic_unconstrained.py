from quadratic_model import x1, y, get_model, train
import numpy as np, statsmodels.formula.api as smf, matplotlib.pyplot as plt, keras.backend as K, pandas as pd

model = get_model(bias_constraint = False, learning_rate = 0.1 * np.sqrt(10))
# model = train(model, epochs = 1000000)
# model.get_weights()
# [array([[-1.0876403 ,  0.85562223]], dtype=float32), array([-3.1043797, -2.7783737], dtype=float32), array([[51.55233],
#        [91.11302]], dtype=float32), array([-6.4673333], dtype=float32)]
# loss ~ 0.0118 # better than the manufactured quadratic fit
model.set_weights([np.array([[-1.0876403 ,  0.85562223]], dtype=np.float32), np.array([-3.1043797, -2.7783737], dtype=np.float32), np.array([[51.55233],
       [91.11302]], dtype=np.float32), np.array([-6.4673333], dtype=np.float32)])

# f1 = sigmoid(-1.0876403*x1 - 3.1043797)
# f2 = sigmoid(0.85562223*x1 - 2.7783737)

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

residuals = y-model.predict(x1).reshape(y.shape)
df = pd.DataFrame({"x1": x1, "f2": f2, "f1": f1, "y": y, "residuals": residuals})
df["x1_2"] = df["x1"]**2
f1_model = smf.ols(formula = 'f1 ~ x1 + x1_2', data = df).fit()
f1_model.summary()
# f1 = 0.0425 -0.0625*x + 0.0246*x^2

f2_model = smf.ols(formula = 'f2 ~ x1 + x1_2', data = df).fit()
f2_model.summary()
# f2 =  0.0580 + 0.0573*x + 0.0190*x^2

np.corrcoef(df['f1'], df['f2'])[1, 0] # not as correlated as the previous model
# -0.5861668506895052
np.corrcoef(df['residuals'], df['f1'])[1, 0]
# -0.014487890787798148
np.corrcoef(df['residuals'], df['f2'])[1, 0]
# 0.008791007726873076

final_layer_model = smf.ols(formula = 'y ~ f1 + f2', data = df).fit()
final_layer_model.summary()
# y = -6.4749 + 51.5308*f1 + 91.1138*f2

model.get_weights()[2:4]
np.corrcoef(final_layer_model.resid, df['f1'])
np.corrcoef(final_layer_model.resid, df['f2'])


# Checking for higher order polynomials in f1, f2
def get_formula_rhs(all_columns, exclude_columns = ['f1', 'f2']):
	formula_columns = list(set(all_columns.tolist()) - set(['y', 'residuals'] + exclude_columns))
	formula_rhs = " + ".join(formula_columns)
	return formula_rhs

def get_summary_df(formula, data):
	model_tmp = smf.ols(formula, data = data).fit()
	rmse = (model_tmp.resid**2).mean()
	summary = model_tmp.summary()
	df = pd.DataFrame(summary.tables[1])
	df = df.iloc[1:, [0, 1, 4]]
	df.columns = ['Variable', 'coef', 'p-value']
	df['coef'] = df['coef'].apply(lambda x: float(str(x)))
	df['p-value'] = df['p-value'].apply(lambda x: float(str(x)))
	return df, rmse

df = pd.DataFrame({"x1": x1, "f2": f2, "f1": f1, "y": y, "residuals": residuals})
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

# Signatures of a cubic polynomial exist in f1 and f2!

plt.scatter(i_s, f1_rmse)
plt.show()

plt.scatter(i_s, f2_rmse)
plt.show()

# What happens to the cubic term in the final function? \hat{y} estimated by the neural network?
f1_cubic_formula = 'f1 ~ x1 + x1_2 + x1_3'
f1_cubic_model_df, _ = get_summary_df(f1_cubic_formula, df)
# f1 = 0.0418 - 0.0477*x1 + 0.0254*x1_2 - 0.0049*x1_3

f2_cubic_formula = 'f2 ~ x1 + x1_2 + x1_3'
f2_cubic_model_df, _ = get_summary_df(f2_cubic_formula, df)
# f2 = 0.0584 + 0.0490*x1 + 0.0185*x1_2 + 0.0028*x1_3

final_layer_model.summary()
# y = -6.4749 + 51.5308*f1 + 91.1138*f2
# Final model approximation of y:

final_y_hat_coefs = 51.5308*f1_cubic_model_df['coef'] + 91.1138*f2_cubic_model_df['coef']
final_y_hat_coefs.iloc[0] += -6.4749
final_y_hat_coefs
# 1    1.000133
# 2    2.006557
# 3    2.994488
# 4    0.002618


# Keeping only the quadratic term:
f1_quadratic_formula = 'f1 ~ x1 + x1_2'
f1_quadratic_model_df, _ = get_summary_df(f1_quadratic_formula, df)
# f1 = 0.0425 - 0.0625*x1 + 0.0246*x1_2

f2_quadratic_formula = 'f2 ~ x1 + x1_2'
f2_quadratic_model_df, _ = get_summary_df(f2_quadratic_formula, df)
# f2 = 0.0580 + 0.0573*x1 + 0.0190*x1_2

final_layer_model.summary()
# y = -6.4749 + 51.5308*f1 + 91.1138*f2
# Final model approximation of y:

final_y_hat_coefs = 51.5308*f1_quadratic_model_df['coef'] + 91.1138*f2_quadratic_model_df['coef']
final_y_hat_coefs.iloc[0] += -6.4749
final_y_hat_coefs
# 1    0.999759
# 2    2.000146
# 3    2.998820
