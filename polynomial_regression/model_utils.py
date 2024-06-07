import statsmodels.formula.api as smf, pandas as pd

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
	df['Variable'] = df['Variable'].astype(str)
	df['coef'] = df['coef'].apply(lambda x: float(str(x)))
	df['p-value'] = df['p-value'].apply(lambda x: float(str(x)))
	return df, rmse


def print_and_subset_summary(model_df, set_variable_index = True):
	print(model_df)
	model_df = model_df[0]
	if set_variable_index:
		model_df = model_df.reset_index(drop = True)
		model_df = model_df.set_index(['Variable'])

	return model_df

