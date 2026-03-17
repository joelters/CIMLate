The goal of CIMLate is to provide a simple to use package to estimate the Average
Treatment Effect (ATE) using different estimators. The package has four user functions:
OLSate, IPWate, directMLate and LRate.

OLSate estimates the ATE by OLS, with and without covariate adjustment.
IPWate estimates the ATE using inverse propensity weighting.
directMLate estimates the ATE from response-function differences.
LRate uses locally robust (orthogonal) scores to implement doubly robust ATE estimation,
with optional cross-fitting and standard errors.

For ML-based estimators, CIMLate now exposes the relevant tuning parameters from
the ML package (polynomial controls, RF/CIF, XGB, CB, Torch, OLSensemble, etc.).
In LRate, response-function (`MLreg`) and propensity-score (`MLps`) models can be tuned
independently via `*.ps` arguments (for example `mtry` vs `mtry.ps`).

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

Examples of the functions are

``` r
n <- 1000
X <- rnorm(n)
D <- runif(n) >= 0.5
Y0 <- X + rnorm(n)
Y1 <- 5 + X + rnorm(n)
Y <- Y1*D + Y0*(1-D)

OLSate(Y,X,D)
IPWate(Y,X,D, ML = "RF")
directMLate(Y,X,D, ML = "RF")
LRate(Y,X,D, MLreg = "RF", MLps = "Logit_lasso")

# Example with different tuning for response and propensity models
LRate(
	Y, X, D,
	MLreg = "RF",
	MLps = "RF",
	mtry = 1,
	mtry.ps = 2,
	rf.cf.ntree = 300,
	rf.cf.ntree.ps = 100
)
```
For more info install the package and see the documentation of the functions with
?OLSate, ?IPWate, ?directMLate and ?LRate.
