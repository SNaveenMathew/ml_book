# 2.1. Linear Regression

## 2.1.1. Model definition

Linear regression model is used to predict a continuous outcome using a set of features. The model is defined as follows:

$$Y = XW + \epsilon$$

For the given sample $(X, y)$ from a non-random (fixed) design we need to estimate the weights $\hat{W}$. The standard way for estimating $W$ is using ordinary least squares (OLS). In this section we will examine OLS as a special case of MLE.

Let us assume that the true model is linear and it's form is known. Additionally, let us assume that the error is normally distributed with 0 mean and constant variance, defined by $\boxed{\epsilon \sim N(0, \sigma^2)}; \sigma^2 = constant$. Let us assume that we have training samples $\boxed{(X^{(i)}, y^{(i)}) \sim_{iid} (X, y)}$. Under these assumptions let us define the *estimates* for the sample as:

$$y = X\hat{W} + e \implies e = y - X \hat{W}$$

Frequentist approach to statistical inference assumes that the true population parameter $W$ is unknown. We are attempting to estimate $\hat{W}$ that closely represents the population parameter $W$. It is important to note that: a) $W$ is fixed and unknown, b) for a given type of estimation on a given data set $\hat{W}$ is fixed. We hope that $\hat{W}$ can be computed and it closely represents the population parameter $W$. Let us use the variable $w$ to denote the 'instantaneous' weight to avoid confusion. Under normality of errors and iid assumptions, we define the likelihood as a function of $w$ given by:

$$L(e; w) = \prod_{i=1}^{N} \bigg[\frac{1}{\sqrt{2 \pi \sigma^2}} exp\bigg(-\frac{(e^{(i)})^2}{2 \sigma^2}\bigg) \bigg]$$

We would like to maximize the likelihood by varying $w$. Since logarithm is a monotonous function of input and likelihood is non-negative (matches the domain of logarithm), the optima (maxima) of $log(L(e; w))$ and the optima (maxima) of $L(e; w)$ are attained at the same point given by $\hat{W}$. We can write the log-likelihood as:

$$log(L(e; w)) = \sum_{i = 1}^{N}\bigg[\frac{1}{\sqrt{2 \pi \sigma^2}} \bigg] + \sum_{i = 1}^{N}\bigg[ \frac{- (e^{(i)})^2}{2 \sigma^2}\bigg]$$

Since $\sigma^2$ is constant, the only term that involves $w$ in $log(L(e; w))$ is the second term. Therefore, we have OLS as a special case of MLE as shown in equation $2.1.1.1$ below:

$$\DeclareMathOperator*{\argmax}{argmax} \DeclareMathOperator*{\argmin}{argmin} \hat{W} = \argmax \limits_{w} log(L(e; w)) = \argmax\limits_{w} \sum_{i=1}^{N}(- e^{(i)})^2 = \argmin\limits_{w} \sum_{i=1}^{N} (e^{(i)})^2 \tag{2.1.1.1}$$

## 2.1.2. Parameter estimation

### 2.1.2.1. Normal equations

Normal equations are obtained by setting the partial derivative of likelihood with respect to $w$ to zero. Since logarithm is a monotonous transformation and likelihood function lies in the domain of logarithm, we can obtain the normal equations by setting the partial derivative of log-likelihood with respect to $w$ to zero.

$$\frac{\partial log(L(e; w))}{\partial w} = 0 \implies \bigg[\frac{\partial}{\partial w} \sum_{i=1}^{N}(e^{(i)})^2 \bigg]_{w = \hat{W}} = 0 \tag{2.1.2.1.1}$$

Using clever linear algebra: $\sum_{i=1}^{N} (e^{(i)})^2 = e^Te$, substituting $e = y - Xw$, using $\frac{\partial e^Te}{\partial w} = 2 \frac{\partial e^T}{\partial w} e$ and substituting $\frac{\partial e^T}{\partial w} = -X^T$ into equation 2.1.2.1.1, we get:

$$ -2 X^T (y - X\hat{W}) = 0 \implies X^Ty - X^TX\hat{W} = 0 \implies X^Ty = (X^TX)\hat{W}$$

$$\implies \hat{W} = (X^TX)^{-1}X^Ty \tag{2.1.2.1.2}$$

#### 2.1.2.1.1. Solving 'simple linear regression'

Simple linear regression is a special case of linear regression in which the number of independent variables is one. The interecept term is almost always included to correct for the residual bias (non-zero mean of the residue), because we assume that the residue accurately models the error term which is normally distributed with 0 mean and constant variance. A simple linear regression model and its corresponding sample esimtates are shown below:

$$Y = X_1 W_1 + W_0 + \epsilon; y = x_1 \hat{W_1} + \hat{W_0} + e$$

For the sample, we can expand the system equation in terms of individual equations given by: $y^{(i)} = x_1^{(i)} \hat{W_1} + \hat{W_0} + e^{(i)}$. Let us write this as a function of weights to solve for $(\hat{W_0}, \hat{W_1}$ that give the OLS solution. We have: $e^{(i)}(w_0, w_1) = (y^{(i)} - x_1^{(i)} w_1 - w_0)$. The OLS loss is given by: $L(w_0, w_1) = \sum_{i=1}^{N_{train}}[e^{(i)}(w_0, w_1)]^2$ with the solution given by $(\hat{W_0}, \hat{W_1}) = \argmin \limits_{(w_0, w_1)} L(w_0, w_1) = \argmin \limits_{(w_0, w_1)} \sum_{i=1}^{N_{train}} [y^{(i)} - x_1^{(i)} w_1 - w_0]^2$. Using the same approach as normal equations, we set the partial derivative of $L(w_0, w_1)$ with respect to $w_0$ and $w_1$ to 0.

$$\bigg[\frac{\partial L(w_0, w_1)}{\partial w_0}\bigg]_{w_1 = \hat{W_1}, w_0 = \hat{W_0}} = \bigg[-2\sum_{i=1}^{N_{train}}(y^{(i)} - x_1^{(i)} w_1 - w_0)\bigg]_{w_1 = \hat{W_1}, w_0 = \hat{W_0}} = 0$$

$$\implies \sum_{i=1}^{N_{train}} e^{(i)} = N_{train} \bar{e} = 0 \tag{2.1.2.1.1.1}$$

$$\implies \frac{1}{N_{train}}\sum_{i=1}^{N_{train}}y^{(i)} = \bar{\hat{y}} = \bar{y} - \bar{e} = \bar{y} = \frac{1}{N_{train}}\sum_{i=1}^{N_{train}}(x_1^{(i)}\hat{W_1} + \hat{W_0}) = \bar{x_1}\hat{W_1} + \hat{W_0} \tag{2.1.2.1.1.2}$$

$$\implies \hat{W_0} = \bar{y} - \hat{W_1}\bar{x_1} \tag{2.1.2.1.1.3}$$

From equation 2.1.2.1.1.1 we infer that the average total residue is equal to zero, or on an average the model does not have a bias. It is because of this property that in deep learning courses we refer to the intercept as the 'bias term'. From equation 2.1.2.1.1.2 we infer that the regression line always passes through the $(\bar{x_1}, \bar{y})$. This is a surprising result and will be used in an analogy in section 2.1.3. Let us continue with the solution:

$$\bigg[\frac{\partial L(w_0, w_1)}{\partial w_1}\bigg]_{w_1 = \hat{W_1}, w_0 = \hat{W_0}} = \bigg[-2\sum_{i=1}^{N_{train}}x_1^{(i)}(y^{(i)} - x_1^{(i)} w_1 - w_0)\bigg]_{w_1 = \hat{W_1}, w_0 = \hat{W_0}} = 0 \tag{2.1.2.1.1.4}$$

$$\implies \sum_{i=1}^{N_{train}} x_1^{(i)} y^{(i)} = \hat{W_1} \sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 + N_{train} \hat{W_0}\bar{x_1} \tag{2.1.2.1.1.5}$$

Substituting equation 2.1.2.1.1.3 in equation 2.1.2.1.1.5:

$$\sum_{i=1}^{N_{train}} x_1^{(i)} y^{(i)} = \hat{W_1} [\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - N_{train}[\bar{x_1}]^2] + N_{train}\bar{x_1}\bar{y}$$

$$\implies \hat{W_1}  = \frac{\sum_{i=1}^{N_{train}} x_1^{(i)} y^{(i)} - N_{train}\bar{x_1}\bar{y}}{\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - N_{train}[\bar{x_1}]^2} \tag{2.1.2.1.1.6}$$

Substituting equation 2.1.2.1.1.6 in 2.1.2.1.1.3:

$$\hat{W_0} = \bar{y} - \frac{\sum_{i=1}^{N_{train}} x_1^{(i)} y^{(i)} - N_{train}\bar{x_1}\bar{y}}{\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - N_{train}[\bar{x_1}]^2}\bar{x_1} = \frac{\bar{y}\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - N_{train}[\bar{x_1}]^2\bar{y} - \bar{x_1}\sum_{i=1}^{N_{train}} x_1^{(i)} y^{(i)} + N_{train}[\bar{x_1}]^2\bar{y}}{\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - N_{train}[\bar{x_1}]^2}$$

$$\implies \hat{W_0} = \frac{\bar{y}\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - \bar{x_1}\sum_{i=1}^{N_{train}} x_1^{(i)} y^{(i)}}{\sum_{i=1}^{N_{train}} [x_1^{(i)}]^2 - N_{train}[\bar{x_1}]^2} \tag{2.1.2.1.1.7}$$

We succeeded in deriving the solution in the hard way. Let's try an easier way:

Substituting equation 2.1.2.1.1.3 in the estimation equation:

$$\hat{y^{(i)}} = \bar{y} + \hat{W_1}(x_1^{(i)} - \bar{x_1}) + e^{(i)}$$

Let us try to estimate the term $\sum_{i=1}^{N_{train}} (y^{(i)}-\bar{y})(x_1^{(i)}-\bar{x_1})$:

$$\sum_{i=1}^{N_{train}} (y^{(i)}-\bar{y})(x_1^{(i)}-\bar{x_1}) = \sum_{i=1}^{N_{train}} \hat{W_1}(x_1^{(i)} - \bar{x_1})^2$$

$$\implies \hat{W_1} = 
\frac{\sum_{i=1}^{N_{train}} (y^{(i)}-\bar{y})(x_1^{(i)}-\bar{x_1})}{\sum_{i=1}^{N_{train}}(x_1^{(i)} - \bar{x_1})^2} \tag{2.1.2.1.1.8}$$

So how did we know the exact term $\sum_{i=1}^{N_{train}} (y^{(i)}-\bar{y})(x_1^{(i)}-\bar{x_1})$? Like most differential equations, without knowing the exact function to substitute, how do we arrive at the solution? Let's examine the term closely: $\sum_{i=1}^{N_{train}} (y^{(i)}-\bar{y})(x_1^{(i)}-\bar{x_1})$. Doesn't this term look familiar? Let's divide it by $N_{train} - 1$: $\frac{\sum_{i=1}^{N_{train}} (y^{(i)}-\bar{y})(x_1^{(i)}-\bar{x_1})}{N_{train} - 1}$. Does it look farmiliar now? Yes, the denominator is $N_{train} - 1$ times the sample covariance. Now guess what the denominator is. Correct, it is $N_{train} - 1$ times the sample variance of $x_1$. Therefore, equation 2.1.2.1.1.8 can be rewritten as:

$$\hat{W_1} = \frac{cov(y, x_1)}{var(x_1)} = \frac{cor(y, x_1) sd(y) sd(x_1)}{[sd(x_1)]^2} = cor(y, x_1)\frac{sd(y)}{sd(x)}$$

#### 2.1.2.1.2. Writing a solver for linear regression

In this section we will try to write code to solve equation 2.1.2.1.2. The first step in building a solver is to write code to multiply two matrices. We use the property that for matrices $A_{n \times m} = [A_{i, j}]_{i = 1, 2, ..., n; j = 1, 2, ..., m}$ and $B_{m \times p} = [B_{j, k}]_{j = 1, 2, ..., m; k = 1, 2, ..., p}$, the product is given by $C_{n \times p} = [C_{i, k}]_{i = 1, 2, ..., n; k = 1, 2, ..., p}$ where $C_{i, k} = \sum_{j = 1}^{m} A_{i, j} * B_{j, k}$. The code can be found below:

[C++ code for multiplying two matrices](data/multiply_matrices.cpp)

The next objective is to invert a matrix. For this purpose we will use row/column reduction, also called Gauss-Jordan elimination or simply Gaussian elimination. We start from the property that for a square matrix $A$ that is invertible (assumed), we have $A^{-1}A = I$. In this property we will rearrange terms in the rows of $A$ to finally arrive at another property given by $A^{-1}I = A^{-1}$.

[C++ code for inverting a matrix](data/invert_matrix.cpp)

The code for finding analytical solution for linear regression can be found in section 2.1.2.2.3. The reader is recommended to go through section 2.1.2.2 to understand the reason for not providing the code here.

#### 2.1.2.1.x. Hierarchy rule

### 2.1.2.2. Computational complexity

Wikipedia defines [computational complexity](https://en.wikipedia.org/wiki/Computational_complexity) of an algorithm as the amount of resources required to run it. Computational complexity of matrix operations can be found in [this article](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra). But how is this calculated?

#### 2.1.2.2.1. Matrix multiplication

Consider two matrices $A_{n \times m}$ and $B_{m \times p}$. The product $C_{n \times p} = AB$ can be computed using the [matrix multiplication code](data/multiply_matrices.cpp). Looking at the loops used in the code (corresponding code block is shown below this paragraph) we find that the total number of operations is of the order $n \times m \times p$. Therefore the complexity (order) of matrix multiplication is $O(nmp)$.

<pre><code>NumericMatrix matrixMultiply(NumericMatrix A, NumericMatrix B) {
  ...
  for(i = 0; i < A.nrow(); i++) {
    for(j = 0; j < B.ncol(); j++) {
      for(k = 0; k < A.ncol(); k++) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  ...
  return(C);
}
</code></pre>

#### 2.1.2.2.1. Matrix inversion

Consider a square matrix $A_{n \times n}$. The inverse $A^{-1}_{n \times n}$ can be computed using the [matrix inversion code](data/invert_matrix.cpp). Looking at the loops used in the code (corresponding code block is shown below this paragraph) we find that the total number of operations is of the order $n \times n \times n$. Therefore the complexity (order) of matrix inversion is $O(n^3)$.

<pre><code>NumericMatrix invertMatrix(NumericMatrix x) {
  ...
  for(i = 0; i < x.nrow(); i++) {
    ...
    for(j = 0; j < x.nrow(); j++) {
      if(j != i) {
        row_num = x(j, i);
        for(k = 0; k < x.nrow(); k++) {
          x(j, k) -= (row_num * x(i, k));
          inv_x(j, k) -= (row_num * inv_x(i, k));
        }
      }
    }
  }
  ...
  return(inv_x);
}
</code></pre>

#### 2.1.2.2.3. Ordinary least squares (linear) regression

Substituting $X: N_{train} \times (p + 1)$, $X^T: (p + 1) \times N_{train}$, $y: N_{train} \times 1$ in equation 2.1.2.1.2, we get the following computations:

- $X^TX: O(N_{train}(p + 1)^2)$
- $(X^TX)^{-1}: O((p + 1)^3)$

Now there are 2 choices for computation:

Choice 1:

- $[(X^TX)^{-1}X^T]: O(N_{train}(p + 1)^2)$
- $[(X^TX)^{-1}X^T]y: O(N_{train}(p + 1))$

Choice 2:

- $[X^Ty]: O(N_{train}(p + 1))$
- $(X^TX)^{-1}[X^Ty]: O((p + 1)^2)$

Addding the total number of computations, we have the computational complexity of the choices as:

- Choice 1: $O(2N_{train}(p + 1)^2 + (p + 1)^3 + N_{train}(p + 1))$
- Choice 2: $O(N_{train}(p + 1)^2 + (p + 1)^2 + (p + 1)^3 + N_{train}(p + 1))$

Comparing the choices we understand that for $N > 1$, $p > 0$ choice 2 is always more optimal. The exercise of finding computational complexity was performed to show that computational linear algebra is not just about mathematics. Careful choices need to be made to design efficient solutions to simple models. The need for performing such an analysis is more evident in more complex models.

[Header file with matrix operations as functions](data/linreg.h)

[C++ code for analytically solving linear regression](data/linreg_analytical.cpp)

The process of finding the total number of computations is the same in complex models such as support vector machines, neural networks, etc. For a linear regression model the choice is straight forward, but for a more complex model the choice may depend on other factors, for example: a) is $p > N_{train}$, b) hyperparameters such as the number of neurons in each layer, c) choice of kernel, etc. Customizing the *solver* may be the difference between training a neural network for years and arriving at the same solution for the same network in minutes. It is simply not enough for data scientists to use `from sklearn.linear_model import LinearRegression`, `lr = LinearRegression()`, and `lr.fit(X_train, y_train)` in Python or `model <- lm(y ~ x, data = train)` in R.

Of course, `sklearn` (Python) and `stats` (R) don't use normal equations because it is computationally expensive. They use methods that find an approximate solution and have much lower computational complexity.

## 2.1.3. Simple linear regression and some physics: an analogy

