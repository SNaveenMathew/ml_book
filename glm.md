# 2. Generalized Linear Models

Generalized linear models (GLMs) cover few of the most commonly used 'machine learning' algorithms in industry: namely linear regression and logistic regression. These algorithms mostly deal with continuous and binary outcomes respectively. GLMs also cover a wide range of algorithms for different types of outcome variables such as: 1) continuous (example: linear regression), 2) count (example: Poisson regression), 3) categorical (example: binary / multinomial / ordinal logistic regression).

In this chapter we will first dive deeper into linear regression. We will sequentially look at other algoritms - binary logistic regression, multinomial/ordinal logistic regression, Poisson regression, GLM with custom link function, etc. For each algorithm we will inspect the following:

1. Model definition
2. Parameter estimation
    - Normal equations
    - Optimization
3. Residual diagnostics
4. Model interpretation

The underlying concept behind estimation of GLM parameters is MLE. The assumption is that all samples $(x^{(i)}, y^{(i)})$ are iid. In special cases parameters can be estimated directly using normal equations (example: a well behaved linear regression). In all other cases parameters can only be estimated using an optimization algorithm that maximizes the likelihood.

2.1. [Linear Regression](linear_regression.md)
2.2. [Logistic Regression](logistic_regression.md)