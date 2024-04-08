The goal of CIMLate is to provide a simple to use package to estimate the Average
Treatment Effect (ATE) using different estimators. The package has three user functions:
OLSate, directMLate and LRate. OLSate estimates the ATE simply by regressing on a treatment
indicator and it also estimates the ATE adjusting for covariates by demeaning the covariates
and interacting them with the treatment indicator. directMLate estimates the conditional expectation
of the outcome given the covariates for the treatment and control groups and then computes the expected
difference among these conditional expectations. The conditional expectations can be estimated by
Lasso, Ridge, Random Forest (RF) or Extreme Gradient Boosting (XGB). LRate uses Locally robust scores
(also known as orthogonal scores) to implement doubly robust estimators of the ATE. Conditional expectations
are estimated as in directMLate and the propensity score can be estimated with RF or Logit Lasso. There is
the option to perform cross-fitting. In the latter case and in the OLS case inference is justified and
standard errors are reported as well. For Lasso, Ridge and Logit Lasso the order of the polynomial to fit
can also be chosen.

## Installation

This package relies on the package ML which you can download from [GitHub](https://github.com/) with:
      
``` r
# install devtools if not installed
install.packages("devtools")
# install ML from github
devtools::install_github("joelters/ML")
```

Then you can install this package with

``` r
# install CIMLate from github
devtools::install_github("joelters/CIMLate")
```

Examples of the three functions are

``` r
n <- 1000
X <- rnorm(n)
D <- runif(n) >= 0.5
Y0 <- X + rnorm(n)
Y1 <- 5 + X + rnorm(n)
Y <- Y1*D + Y0*(1-D)

OLSate(Y,X,D)
directMLate(Y,X,D, ML = "RF")
LRate(Y,X,D, MLreg = "RF", MLps = "Logit_lasso")
```
For more info install the package and see the documentation of the functions with
?OLSate, ?directMLate and ?LRate. For now it is not possible to change the tuning parameters
in the Machine Learning algoriothms (they are set to the default parameters).
If needed I suggest using the trace() function to change the tuning parameters
(see [here](https://stackoverflow.com/questions/34800331/r-modify-and-rebuild-package)).
