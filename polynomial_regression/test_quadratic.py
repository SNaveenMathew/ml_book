from quadratic_model import x1, y, get_model, train, x1_extended, y_extended
import numpy as np, pandas as pd, statsmodels.formula.api as smf, matplotlib.pyplot as plt, keras.backend as K, pickle
from plot_predictions import plot_pred_matrix
from model_utils import get_formula_rhs, get_summary_df, print_and_subset_summary

model = get_model()
model.get_weights()
# loss ~ 0.0458
# [array([[0.1920907 , 0.39586592]], dtype=float32), array([ 0.      , -1.316958], dtype=float32), array([[-549.6529 ],
#        [ 427.67603]], dtype=float32), array([185.34625], dtype=float32)]
model.set_weights([\
	np.array([[0.1920907 , 0.39586592]], dtype=np.float32), np.array([ 0.      , -1.316958], dtype=np.float32), np.array([[-549.6529 ],\
        [ 427.67603]], dtype=np.float32), np.array([185.34625], dtype=np.float32)
])
# If you are not satisfied with this solution:
# model = train(model, epochs = 5000000) # Takes a long time to train
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

plt.scatter(i_s, f1_rmse)
plt.show()

plt.scatter(i_s, f2_rmse)
plt.show()


residuals = y-model.predict(x1).reshape(y.shape)
model.get_weights()[0:2]
df = pd.DataFrame({"x1": x1, "f2": f2, "f1": f1, "y": y, "redisuals": residuals})
df["x1_2"] = df["x1"]**2
linear_term_model = smf.ols(formula = 'f1 ~ x1', data = df).fit()
linear_term_model.summary()
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.5000   3.53e-06   1.42e+05      0.000       0.500       0.500
# x1             0.0476   3.54e-06   1.35e+04      0.000       0.048       0.048
# ==============================================================================

# Theoretical
model.get_weights()[0][0][0] * np.array([0.19, 0.25]) # d/dx (sigmoid(x)) in the neighborhood of x = 0 \in (0.19 0.25)
# array([0.03649723, 0.04802268])

model.get_weights()[0:2]
quadratic_term_model = smf.ols(formula = 'f2 ~ x1 + x1_2', data = df).fit()
quadratic_term_model.summary()
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.2116   6.71e-06   3.15e+04      0.000       0.212       0.212
# x1             0.0658    5.5e-06    1.2e+04      0.000       0.066       0.066
# x1_2           0.0070   3.87e-06   1796.986      0.000       0.007       0.007
# ==============================================================================

# Theoretical
model.get_weights()[0][0][1]**2 * 0.0455 # d/dx (sigmoid(x)) at x = log(2 - sqrt(3)) ~ 0.0455
# 0.007130297010436493


df = df.sort_values(["x1"])
df["df1"] = df["f1"] - pd.Series([np.nan] + df["f1"].iloc[:-1].tolist())
df["dx1"] = df["x1"] - pd.Series([np.nan] + df["x1"].iloc[:-1].tolist())
df["df1_dx1"] = df["df1"] / df["dx1"]
plt.scatter(df["x1"], df["df1_dx1"])
plt.show()

df["df2"] = df["f2"] - pd.Series([np.nan] + df["f2"].iloc[:-1].tolist())
df["df2_dx1"] = df["df2"] / df["dx1"]
plt.scatter(df["x1"], df["df2_dx1"])
plt.show()

df["d2f2"] = df["df2"] - pd.Series([np.nan] + df["df2"].iloc[:-1].tolist())
df["d2f2_dx12"] = df["d2f2"] / df["dx1"]
plt.scatter(df["x1"], df["d2f2_dx12"])
plt.show()

df["d2f2_dx12"].median()
# 0.09727161309447503

df["d2f2_dx12"].mean()
# 0.11184993246848307

tmp = df["d2f2_dx12"].abs()
tmp[tmp < 1].plot.hist(bins = 100)
plt.show()

final_layer_model = smf.ols(formula = 'y ~ f1 + f2', data = df).fit()
final_layer_model.summary()
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    185.8187      0.105   1769.029      0.000     185.613     186.025
# f1          -551.0261      0.303  -1820.782      0.000    -551.619    -550.433
# f2           428.6562      0.215   1990.837      0.000     428.234     429.078
# ==============================================================================
model.get_weights()[2:4]
# [array([[-549.6529 ],
#        [ 427.67603]], dtype=float32), array([185.34625], dtype=float32)]


residuals_model = smf.ols(formula = 'residuals ~ f1 + f2', data = df).fit()
residuals_model.summary() # Intercept, f1 and f2 all are insignificant
np.corrcoef(residuals, f1)[1, 0]
# -0.00243
np.corrcoef(residuals, f2)[1, 0]
# 0.00436
np.corrcoef(f1, f2)[1, 0] # Highly correlated, variance inflation may be a concern for the model
# 0.9889

final_layer_model.params[2] * quadratic_term_model.params[2]
# 2.979394429793285

final_layer_model.params[2] * quadratic_term_model.params[1] + final_layer_model.params[1] * linear_term_model.params[1]
# 2.0014364856788696

final_layer_model.params[0] + final_layer_model.params[2] * quadratic_term_model.params[0] + final_layer_model.params[1] * linear_term_model.params[0]
# 1.0205434934664481
