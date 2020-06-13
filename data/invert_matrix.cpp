#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix invertMatrix(NumericMatrix x) {
  // Initializing inv_x to identity matrix
  NumericMatrix inv_x(x.nrow(), x.ncol());
  float row_div, row_num;
  int i = 0, j = 0, k = 0;
  for(i = 0; i < x.nrow(); i++) {
    inv_x(i, i) = 1;
  }
  
  for(i = 0; i < x.nrow(); i++) {
    // Making x(i, i) = 1 by dividing whole row by x(i, i)
    // If x(i, i) = 0 we add a suitable row to make it non-zero
    if(x(i, i) == 0) {
      for(j = i + 1; j < x.nrow(); j++) {
        if(x(j, i) != 0) {
          for(k = 0; j < x.nrow(); k++) {
            x(i, k) += x(j, k);
            inv_x(i, k) += inv_x(j, k);
          }
        }
      }
    }
    row_div = x(i, i);
    for(j = 0; j < x.nrow(); j++) {
      x(i, j) /= row_div;
      inv_x(i, j) /= row_div;
    }
    
    // Performing row operations to have x(i, i) = 1; x(j, i) = 0 if j \neq i
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
  return(inv_x);
}


/*** R
x <- 1:9
x[9] <- 10 # Making it invertible
mat <- matrix(x, nrow = 3, ncol = 3, byrow = TRUE)
solve(mat)
invertMatrix(mat)
*/
