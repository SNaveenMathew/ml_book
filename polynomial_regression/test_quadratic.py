from quadratic_model import x1, y, get_model, train
import numpy as np, pandas as pd, statsmodels.formula.api as smf, matplotlib.pyplot as plt, keras.backend as K, pandas as pd

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

residuals = y-model.predict(x1).reshape(y.shape)
model.get_weights()[0:2]
df = pd.DataFrame({"x1": x1, "f2": f2, "f1": f1, "y": y, "redisuals": residuals})
df["x1_2"] = df["x1"]**2
linear_term_model = smf.ols(formula = 'f1 ~ x1', data = df).fit()
linear_term_model.summary()
model.get_weights()[0][0][0] * np.array([0.19, 0.25]) # d/dx (sigmoid(x)) in the neighborhood of x = 0 \in (0.19 0.25)

model.get_weights()[0:2]
quadratic_term_model = smf.ols(formula = 'f2 ~ x1 + x1_2', data = df).fit()
quadratic_term_model.summary()
model.get_weights()[0][0][1]**2 * 0.0455 # d/dx (sigmoid(x)) at x = log(2 - sqrt(3)) ~ 0.0455


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

tmp = df["d2f2_dx12"].abs()
tmp[tmp < 1].plot.hist(bins = 100)
plt.show()

final_layer_model = smf.ols(formula = 'y ~ f1 + f2', data = df).fit()
final_layer_model.summary()
model.get_weights()[2:4]

residuals_model = smf.ols(formula = 'residuals ~ f1 + f2', data = df).fit()
residuals_model.summary() # Intercept, f1 and f2 all are insignificant
np.corrcoef(residuals, f1)
np.corrcoef(residuals, f2)
np.corrcoef(f1, f2) # Highly correlated, variance inflation may be a concern for the model
