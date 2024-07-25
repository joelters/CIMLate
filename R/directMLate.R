#' Estimate Average Treatment Effect (ATE) using machine learning to
#' estimate the conditional expectations
#'
#' `directMLate` estimates the ATE by estimating the conditional expectation
#' of Y given X for the treated: mu_1(X) and for the control: mu_0(X) and
#' estimating the expected difference among them.
#'
#'
#' @param X is a dataframe containing all the covariates
#' @param Y is a vector containing the outcome
#' @param D the treatment indicator
#' @param ML is a string specifying which machine learner to use among
#' Lasso, Ridge, Random Forests or XGBoosting
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
#' directMLate(Y,X,D, ML = "RF")
#'
#' @export
directMLate <- function(Y,X,D, ML = c("Lasso", "Ridge", "RF", "XGB","grf"), polynomial = 1){

  if (!("data.frame" %in% class(X))){
    X <- data.frame(X)
  }

  mu1 <- ML::modest(X[D == 1,], Y[D == 1], ML = ML, polynomial = polynomial)
  mu1 <- ML::FVest(mu1,X,Y,Xnew = X, Ynew = Y, ML = ML, polynomial = polynomial)
  mu0 <- ML::modest(X[D == 0,], Y[D == 0], ML = ML, polynomial = polynomial)
  mu0 <- ML::FVest(mu0,X,Y,Xnew = X, Ynew = Y, ML = ML, polynomial = polynomial)


  b3 <- round(mean(mu1 - mu0),3)
  resdirect <- data.frame("ATE" = b3,"ML" = ML)
  return(resdirect)
}
