#' Estimate Average Treatment Effect (ATE) using the Inverse Propensity Weighting
#' method and machine learning to estimate the propensity score
#'
#' `IPWate` estimates the ATE by estimating the propensity score by Machine Learning
#' and implementing a plug-in IPW estimator
#'
#'
#' @param X is a dataframe containing all the covariates
#' @param Y is a vector containing the outcome
#' @param D the treatment indicator
#' @param ML is a string specifying which machine learner to use among
#' Random Forests or Lasso logit
#' @param pscore vector with treatment probabilities if known (e.g. in an RCT).
#' Otherwise leave NULL so that they are estimated using MLps
#' @param polynomial degree of polynomial to be fitted when using Lasso, Ridge
#' or Logit Lasso. 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @returns dataframe with estimates
#' @examples
#' n <- 1000
#' X <- rnorm(n)
#' D <- runif(n) >= 0.5
#' Y0 <- X + rnorm(n)
#' Y1 <- 5 + X + 5*X + rnorm(n)
#' Y <- Y1*D + Y0*(1-D)
#' IPWate(Y,X,D, ML = "RF")
#'
#' @export
IPWate <- function(Y,X,D, ML = c("RF", "Lasso_logit","grf"), pscore = NULL, polynomial = 1){

  if (!("data.frame" %in% class(X))){
    X <- data.frame(X)
  }
  if (is.null(pscore)){
    ps <- ML::MLest(X,D,ML = ML, polynomial = polynomial)
    ps <- ps$FVs
  }
  else {
    ps <- pscore
  }
  if (sum(ps <= 0 | ps >= 1) > 0){
    warning("There are estimated propensity scores <= 0 or >= 1,
                they have been changed to 0.001 or 0.999")
    ps <- ps*(ps > 0 | ps < 1) +
      0.001*(ps <= 0) + 0.999*(ps >= 1)
  }

  b3 <- round(mean(Y*D/ps - Y*(1-D)/(1-ps)),3)
  resdirect <- data.frame("ATE" = b3,"ML" = ML)
  return(resdirect)
}
