#' Estimate Average Treatment Effect (ATE) in Randomized Control Trials (RCTs) with OLS
#'
#' `OLSate` estimates the ATE with data from RCTs. Reports both the estimate which
#' is unadjusted by covariates and the estimate adjusted by covariates. The adjustment
#' is made recentering the covariates and interacting them with the treatment so that
#' precision is improved even if the model is not correctly specified.
#'
#'
#' @param X is a dataframe containing all the covariates
#' @param Y is a vector containing the outcome
#' @param D the treatment indicator
#' @returns dataframe with estimates and standard errors
#' @examples
#' n <- 1000
#' X <- rnorm(n)
#' D <- runif(n) >= 0.5
#' Y0 <- X + rnorm(n)
#' Y1 <- 5 + X + rnorm(n)
#' Y <- Y1*D + Y0*(1-D)
#' OLSate(Y,X,D)
#'
#' @export
OLSate <- function(Y,X,D){
  dff <- data.frame(Y = Y, D = D, X)
  # Regression
  m1 <- stats::lm(Y ~ D, dff)
  vcv <- sandwich::vcovHC(m1, type = "HC1")
  res1 <- lmtest::coeftest(m1, vcv)
  b1 <- round(res1[2,1],3)
  se1 <- round(res1[2,2],3)
  resols <- data.frame("ATE" = b1, "se" = se1,"Method" = "OLS")

  # Adjusted Regression
  Xc <- scale(X, center = TRUE, scale = FALSE)
  dffc <- data.frame(Y = Y, D = D, Xc)

  m2 <- stats::lm(Y ~ . + D:(.), dffc)
  vcv <- sandwich::vcovHC(m2, type = "HC1")
  res2 <- lmtest::coeftest(m2, vcv)
  b2 <- round(res2[2,1],3)
  se2 <- round(res2[2,2],3)
  resolsadj <- data.frame("ATE" = b2, "se" = se2,"Method" = "OLS adjusted")
  res <- rbind(resols, resolsadj)
  return(res)
}
