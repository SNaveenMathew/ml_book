#include <Rcpp.h>
#include <random>
#include "linreg.h"
using namespace Rcpp;

NumericMatrix grad_W(NumericMatrix x, NumericMatrix e) {
  NumericMatrix grad(x.ncol(), 1), xT(x.ncol(), x.nrow());
  int i = 0;
  xT = transpose(x);
  grad = matrixMultiply(xT, e);
  for(i = 0; i < grad.nrow(); i++) {
    grad(i, 0) = -2 * grad(i, 0);
  }
  return(grad);
}

NumericMatrix get_hessian(NumericMatrix x) {
  NumericMatrix xT(x.ncol(), x.nrow()), h(x.ncol(), x.ncol());
  xT = transpose(x);
  int i = 0, j = 0;
  h = matrixMultiply(xT, x);
  for(i = 0; i < x.nrow(); i++) {
    for(j = 0; j < x.ncol(); j++) {
      h(i, j) = 2 * h(i, j);
    }
  }
  return(h);
}

double get_loss(NumericMatrix e) {
  double loss = 0;
  int i = 0;
  for(i = 0; i < e.nrow(); i++) {
    loss += e(i, 0) * e(i, 0);
  }
  return(loss);
}

double double_abs(double x) {
  if(x < 0) {
    return(-x);
  } else {
    return(x);
  }
}

NumericMatrix get_resid(NumericMatrix y, NumericMatrix yhat) {
  NumericMatrix e(y.nrow(), 1);
  int j = 0;
  for(j = 0; j < y.nrow(); j++) {
    e(j, 0) = y(j, 0) - yhat(j, 0);
  }
  return(e);
}

// [[Rcpp::export]]
NumericMatrix linreg_Newton(NumericMatrix x, NumericMatrix y, int max_iter = 100,
                            double gamma = 1, double loss_tol = 1e-6) {
  NumericMatrix w(x.ncol(), 1), grad(x.ncol(), 1), w_next(x.ncol(), 1), e(y.nrow(), 1), yhat(y.nrow(), 1), hessian(x.ncol(), x.ncol()),  inv_hessian(x.ncol(), x.ncol()), inv_hessian_jacobian(x.ncol(), 1);
  hessian = get_hessian(x);
  inv_hessian = invertMatrix(hessian); // constant throughout
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 0.02);
  double tol = 1, loss_p, loss_n;
  int iter = 0, j = 0;
  for(j = 0; j < w.nrow(); j++) {
    w(j, 0) = distribution(generator);
  }
  while(iter < max_iter && tol > loss_tol) {
    yhat = matrixMultiply(x, w);
    e = get_resid(y, yhat);
    loss_p = get_loss(e);
    grad = grad_W(x, e);
    inv_hessian_jacobian = matrixMultiply(inv_hessian, grad);
    for(j = 0; j < w_next.nrow(); j++) {
      w_next(j, 0) = w(j, 0) - gamma * inv_hessian_jacobian(j, 0);
    }
    yhat = matrixMultiply(x, w_next);
    e = get_resid(y, yhat);
    loss_n = get_loss(e);
    tol = (loss_p - loss_n)/loss_p;
    tol = double_abs(tol);
    w = w_next;
    iter++;
  }
  if(iter < max_iter) {
    printf("Newton's method converged in %d iterations for relative loss tolerance = %lf", iter, loss_tol);
  } else {
    printf("Newton's method did not converge in %d iterations for relative loss tolerance = %lf", max_iter, loss_tol);
  }
  return(w);
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
# Normal equations
solve(t(x1) %*% x1) %*% (t(x1) %*% y1)
# Newton's method
linreg_Newton(x = x1, y = y1)

# Compare with lm function
model <- lm(y ~ x)
model$coefficients

*/
