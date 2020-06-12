# 2.1. Linear Regression

## 2.1.1. Model definition

Linear regression model is used to predict a continuous outcome using a set of features. The model is defined as follows:

\\(Y = XW + \\epsilon\\)

For the given sample $(X, y)$ from a non-random (fixed) design we need to estimate the weights $\hat{W}$. The standard way for estimating W is using ordinary least squares (OLS). In this section we will examine OLS as a special case of MLE.

Let us assume that the true model is linear and it's form is known. Additionally, let us assume that the error is normally distributed with 0 mean and constant variance, defined by $\epsilon \sim N(0, \sigma^2); \sigma^2 = constant$. Let us assume that we have samples $(X^{(i)}, y^{(i)}) \sim_{iid} (X, y)$. Under these assumptions let us define the estimates for the sample as:

$$y = X\hat{W} + e; e = y - X \hat{W}$$

Under normality of errors and iid assumptions, we define the likelihood as:

$$L(e; W) = \prod_{i=1}^{N} \bigg[\frac{1}{\sqrt{2 \pi \sigma^2}} exp\bigg(-\frac{(e^{(i)})^2}{2 \sigma^2}\bigg) \bigg]$$

We would like to maximize the likelihood by varying W. Since logarithm is a monotonous function of input and likelihood is non-negative (matches the domain of logarithm), the optima (maxima) of $log(L(e; W))$ and the optima (maxima) of $L(e; W)$ are attained at the same point given by $\hat{W}$. We can write the log-likelihood as:

$$log(L(e; W)) = \sum_{i = 1}^{N}\bigg[\frac{1}{\sqrt{2 \pi \sigma^2}} \bigg] + \sum_{i = 1}^{N}\bigg[ \frac{- (e^{(i)})^2}{2 \sigma^2}\bigg]$$

Since $\sigma^2$ is constant, the only term that involves W in $log(L(e; W))$ is the second term. Therefore, we have the OLS as a special case of MLE as:

$$\hat{W} = argmax_W log(L(e; W)) = argmax_W \sum_{i=1}^{N}(- e^{(i)})^2 = argmin_W \sum_{i=1}^{N} (e^{(i)})^2$$

## 2.1.2. Parameter estimation

### 2.1.2.1. Normal equations

Normal equations are obtained by setting the partial derivative of likelihood with respect to W to zero. Since logarithm is a monotonous transformation and likelihood function lies in the domain of logarithm, we can obtain the normal equations by setting the partial derivative of log-likelihood with respect to W to zero.

\begin{equation}
\frac{\partial log(L(e; W))}{\partial W} = 0 \implies \bigg[\frac{\partial}{\partial W} \sum_{i=1}^{N}(e^{(i)})^2 \bigg]_{W = \hat{W}} = 0
\end{equation}

Using clever linear algebra: $\sum_{i=1}^{N} (e^{(i)})^2 = e^Te$, substituting $e = y - X\hat{W}$, using $\frac{\partial e^Te}{\partial W} = 2 \frac{\partial e}{\partial W} e$ and substituting $\frac{\partial e}{\partial W} = X^T$, we get:

$$ 2 * X^T (y - X\hat{W}) = 0 \implies X^Ty - X^TX\hat{W} = 0 \implies X^Ty = (X^TX)\hat{W} \implies \hat{W} = (X^TX)^{-1}X^Ty$$