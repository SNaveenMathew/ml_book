# Notation

Terminology:

- Features: Independent variables used for predictive modeling
- Outcome: Dependent variable
- Feature matrix: Matrix of independent variables with samples along rows and features along columns
- Example: One sample (row) from feature matrix

Notation:

- $X$: Feature matrix containing several samples and one or more independent variables
- $y$: Vector of outcomes - can sometimes be a one-hot matrix
- $X^{(i)}$: $i^{th}$ sample/example/row in matrix X; matrix X includes a column with value 1 throughout to accommodate the intercept term in weights unless specified otherwise
- $y^{(i)}$: $i^{th}$ value in vector y
- $W$: Weight matrix of true model (unknown); includes the intercept unless specified otherwise
- $\hat{W}$: Estimated weight matrix of the model
- $N_{train}$: Number of samples in training set
- var: Variance/covariance operator on a series or matrix
- sd: Standard deviation operator on a series
- E: Expectation operator on a series
- $X^TX$: Gram matrix (or covariance matrix if X is centered) of feature matrix X

Abbreviations:

- iid: Independent, identically distributed
