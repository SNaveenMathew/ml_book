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

// Can probably do better!
NumericMatrix get_mat_without_column(NumericMatrix x, int col_num) {
  NumericMatrix result(x.nrow(), x.ncol()-1);
  int j = 0, x_col = 0, i = 0;
  while(j < result.ncol()) {
    if(x_col != col_num) {
      for(i=0; i<x.nrow(); i++) {
        result(i, j) = x(x_col, j);
      }
      j++;
    }
    x_col++;
  }
  return(result);
}

// Can probably do better!
NumericMatrix get_mat_column(NumericMatrix x, int col_num) {
  NumericMatrix result(x.nrow(), 1);
  int i = 0;
  for(i=0; i<x.nrow(); i++) {
    result(i, 0) = x(i, col_num);
  }
  return(result);
}

// Can probably do better!
NumericMatrix get_vec_without_row(NumericMatrix w, int row_num) {
  NumericMatrix result(w.nrow() - 1, 1);
  int j = 0, w_row = 0;
  while(j < result.ncol()) {
    if(w_row != row_num) {
      result(j, 0) = w(w_row, 0);
      j++;
    }
    w_row++;
  }
  return(result);
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

double sum_squares(NumericMatrix x_col) {
  int j = 0;
  double res = 0;
  for(j = 0; j < x_col.nrow(); j++) {
    res += x_col(j, 0) * x_col(j, 0);
  }
  return(res);
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
NumericMatrix linreg_coord(NumericMatrix x, NumericMatrix y, int max_iter = 1,
                           double gamma = 1, double loss_tol = 1e-6) {
  NumericMatrix w(x.ncol(), 1), grad(x.ncol(), 1), w_next(x.ncol(), 1), e(y.nrow(), 1), yhat(y.nrow(), 1), hessian(x.ncol(), x.ncol()),  inv_hessian(x.ncol(), x.ncol()), inv_hessian_jacobian(x.ncol(), 1), x_j(x.nrow(), 1), x_jT(1, x.nrow()), x_jTyhat(1, 1), remaining_x(x.nrow(), x.ncol()-1), remaining_w(x.ncol()-1, 1);
  double sum_sq_x_j;
  hessian = get_hessian(x);
  inv_hessian = invertMatrix(hessian); // constant throughout
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 0.0);
  double tol = 1, loss_p, loss_n;
  int iter = 0, j = 0;
  for(j = 0; j < w.nrow(); j++) {
    w(j, 0) = distribution(generator);
    w_next(j, 0) = distribution(generator);
  }
  while(iter < max_iter && tol > loss_tol) {
    yhat = matrixMultiply(x, w);
    e = get_resid(y, yhat);
    loss_p = get_loss(e);
    for(j = 0; j < w_next.nrow(); j++) {
      remaining_x = get_mat_without_column(x, j);
      remaining_w = get_vec_without_row(w_next, j);
      yhat = matrixMultiply(x, w_next);
      x_j = get_mat_column(x, j);
      x_jT = transpose(x_j);
      x_jTyhat = matrixMultiply(x_jT, yhat);
      sum_sq_x_j = sum_squares(x_j); //Same as matrixMultiply(x_jT, x_j)(0, 0);
      w_next(j, 0) = x_jTyhat(0, 0)/sum_sq_x_j;
    }
    // printf("%lf", w_next(0, 0));
    
    yhat = matrixMultiply(x, w_next);
    e = get_resid(y, yhat);
    loss_n = get_loss(e);
    tol = (loss_p - loss_n)/loss_p;
    tol = double_abs(tol);
    w = w_next;
    iter++;
  }
  if(iter < max_iter) {
    printf("Coordinate descent converged in %d iterations for relative loss tolerance = %lf", iter, loss_tol);
  } else {
    printf("Coordinate descent did not converge in %d iterations for relative loss tolerance = %lf", max_iter, loss_tol);
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
linreg_coord(x = x1, y = y1)

# Compare with lm function
model <- lm(y ~ x)
model$coefficients

*/
