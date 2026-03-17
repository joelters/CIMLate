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
#' Random Forests, CIF, XGB, CB, Logit Lasso, grf or SuperLearner
#' @param pscore vector with treatment probabilities if known (e.g. in an RCT).
#' Otherwise leave NULL so that they are estimated using ML
#' @param SL.library string vector with learners to use in SuperLearner
#' @param polynomial.Logit_lasso degree of polynomial to be fitted when using
#' Logit Lasso. 1 just fits the input X. 2 squares all variables and adds all
#' pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
#' @param polynomial.Lasso degree of polynomial to be fitted when using Lasso,
#' see polynomial.Logit_lasso for more info
#' @param polynomial.Ridge degree of polynomial to be fitted when using Ridge,
#' see polynomial.Logit_lasso for more info
#' @param polynomial.OLS degree of polynomial to be fitted when using OLS,
#' see polynomial.Logit_lasso for more info
#' @param polynomial.NLLS_exp degree of polynomial to be fitted when using
#' NLLS_exp, see polynomial.Logit_lasso for more info
#' @param polynomial.loglin degree of polynomial to be fitted when using
#' loglin, see polynomial.Logit_lasso for more info
#' @param rf.cf.ntree how many trees should be grown when using RF or CIF
#' @param rf.depth how deep should trees be grown in RF (NULL is full depth,
#' the default from ranger)
#' @param mtry number of variables to possibly split at in each node for RF or CIF
#' @param cf.depth how deep should trees be grown in CIF (Inf is default from partykit)
#' @param xgb.nrounds number of boosting rounds to use in XGB
#' @param xgb.max.depth maximum tree depth in XGB
#' @param cb.iterations maximum number of trees that can be built in CB
#' @param cb.depth depth of trees in CB
#' @param torch.epochs number of training epochs for the Torch neural network
#' @param torch.hidden_units numeric vector specifying the number of neurons in
#' each hidden layer of the Torch neural network
#' @param torch.lr learning rate for the Torch optimizer
#' @param torch.dropout dropout rate for regularization in the Torch neural
#' network (between 0 and 1)
#' @param intercept logical; should an intercept be included? Default TRUE.
#' Only applies to OLS, Lasso, Ridge, and loglin.
#' @param csplot logical; if TRUE, returns a histogram overlay of propensity
#' scores by treatment status to assess common support
#' @returns dataframe with estimates, or a list with `results` and `csplot`
#' when `csplot = TRUE`
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
IPWate <- function(Y, X, D,
                   ML = c("RF", "Logit_lasso", "CIF", "XGB", "CB", "grf", "SL"),
                   pscore = NULL,
                   SL.library = c("SL.ranger"),
                   polynomial.Logit_lasso = 1,
                   polynomial.Lasso = 1,
                   polynomial.Ridge = 1,
                   polynomial.OLS = 1,
                   polynomial.NLLS_exp = 1,
                   polynomial.loglin = 1,
                   rf.cf.ntree = 500,
                   rf.depth = NULL,
                   mtry = max(floor(ncol(X)/3), 1),
                   cf.depth = Inf,
                   xgb.nrounds = 200,
                   xgb.max.depth = 6,
                   cb.iterations = 500,
                   cb.depth = 6,
                   torch.epochs = 50,
                   torch.hidden_units = c(64, 32),
                   torch.lr = 0.01,
                   torch.dropout = 0.2,
                   intercept = TRUE,
                   csplot = FALSE) {

  if (!("data.frame" %in% class(X))) {
    X <- data.frame(X)
  }
  if (is.null(pscore)) {
    ps <- ML::MLest(X, D, ML = ML,
                    SL.library = SL.library,
                    polynomial.Logit_lasso = polynomial.Logit_lasso,
                    polynomial.Lasso = polynomial.Lasso,
                    polynomial.Ridge = polynomial.Ridge,
                    polynomial.OLS = polynomial.OLS,
                    polynomial.NLLS_exp = polynomial.NLLS_exp,
                    polynomial.loglin = polynomial.loglin,
                    rf.cf.ntree = rf.cf.ntree,
                    rf.depth = rf.depth,
                    mtry = mtry,
                    cf.depth = cf.depth,
                    xgb.nrounds = xgb.nrounds,
                    xgb.max.depth = xgb.max.depth,
                    cb.iterations = cb.iterations,
                    cb.depth = cb.depth,
                    torch.epochs = torch.epochs,
                    torch.hidden_units = torch.hidden_units,
                    torch.lr = torch.lr,
                    torch.dropout = torch.dropout,
                    intercept = intercept)
    ps <- ps$FVs
  } else {
    ps <- pscore
  }
  if (sum(ps <= 0 | ps >= 1) > 0) {
    warning("There are estimated propensity scores <= 0 or >= 1,
                they have been changed to 0.001 or 0.999")
    ps <- pmin(pmax(ps, 0.001), 0.999)
  }

  b3 <- round(mean(Y * D / ps - Y * (1 - D) / (1 - ps)), 3)
  resdirect <- data.frame("ATE" = b3, "ML" = ML)
  if (csplot) {
    p <- csplot_common_support(ps = ps, D = D)
    return(list(results = resdirect, csplot = p))
  }
  return(resdirect)
}
