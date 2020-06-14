#include <Rcpp.h>
#include "linreg.h"
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix linreg(NumericMatrix x, NumericMatrix y) {
  NumericMatrix xT(x.ncol(), x.nrow()), xT_x(x.ncol(), x.ncol()), xT_x_inv(x.ncol(), x.ncol()), xT_y(x.ncol(), 1), W(x.ncol(), 1);
  xT = transpose(x);
  xT_x = matrixMultiply(xT, x);
  xT_x_inv = invertMatrix(xT_x);
  xT_y = matrixMultiply(xT, y);
  W = matrixMultiply(xT_x_inv, xT_y);
  return(W);
}

/*** R
set.seed(1)
n_train <- 100
x <- 1:n_train
error_sd <- 0.5
res <- rnorm(n = length(x), mean = 0, sd = error_sd)
w_0 <- 0
w_1 <- 2
y <- w_0 + w_1 * x + res

x1 <- cbind(1, x)
y1 <- matrix(y, nrow = length(y), ncol = 1)
solve(t(x1) %*% x1) %*% (t(x1) %*% y1)
linreg(x = x1, y = y1)

# Compare with lm function
model <- lm(y ~ x)
model$coefficients

*/
