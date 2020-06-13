#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix matrixMultiply(NumericMatrix A, NumericMatrix B) {
  NumericMatrix C(A.nrow(), B.ncol());
  int i = 0, j = 0, k = 0;
  for(i = 0; i < A.nrow(); i++) {
    for(j = 0; j < B.ncol(); j++) {
      for(k = 0; k < A.ncol(); k++) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  return(C);
}

/*** R
x <- 1:8
mat1 <- matrix(x, nrow = 2, ncol = 4, byrow = T)
mat2 <- matrix(x, nrow = 4, ncol = 2, byrow = T)
mat1 %*% mat2
matrixMultiply(A = mat1, B = mat2)

mat2 %*% mat1
matrixMultiply(A = mat2, B = mat1)
*/
