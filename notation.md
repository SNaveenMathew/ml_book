# Notation

Terminology:

- Features: Independent variables used for predictive modeling
- Outcome: Dependent variable
- Feature matrix: Matrix of independent variables with samples along columns and features along rows
- Example: One sample (column) from feature matrix

Notation:

- X: Feature matrix containing several samples and one or more independent variables
- y: Vector of outcomes - can sometimes be a one-hot matrix
- $X^{(i)}$: $i^{th}$ sample/example/column in matrix X; matrix X includes a column with value 1 throughout to accommodate the intercept term in weights unless specified otherwise
- $y^{(i)}$: $i^{th}$ value in vector y
- W: Weight matrix of true model (unknown); includes the intercept unless specified otherwise
- $\hat{W}$: Estimated weight matrix of the model

Abbreviations:

- iid: Independent, identically distributed

