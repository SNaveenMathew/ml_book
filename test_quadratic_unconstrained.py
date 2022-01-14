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
df = pd.DataFrame({"x1": x1, "f2": f2, "f1": f1, "y": y, "redisuals": residuals})
df["x1_2"] = df["x1"]**2
f1_model = smf.ols(formula = 'f1 ~ x1 + x1_2', data = df).fit()
f1_model.summary()
# f1 = 0.0425 -0.0625*x + 0.0246*x^2

f2_model = smf.ols(formula = 'f2 ~ x1 + x1_2', data = df).fit()
f2_model.summary()
# f2 =  0.0580 + 0.0573*x + 0.0190*x^2

np.corrcoef(f1, f2) # not as correlated as the previous model
np.corrcoef(residuals, f1)
np.corrcoef(residuals, f2)

final_layer_model = smf.ols(formula = 'y ~ f1 + f2', data = df).fit()
final_layer_model.summary()
model.get_weights()[2:4]
np.corrcoef(final_layer_model.resid, df['f1'])
np.corrcoef(final_layer_model.resid, df['f2'])
