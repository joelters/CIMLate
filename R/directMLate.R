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
#' Lasso, Ridge, Random Forests, CIF, XGBoosting, Catboosting, Torch,
#' grf, OLSensemble or SuperLearner
#' @param SL.library string vector with learners to use in SuperLearner
#' @param polynomial.Lasso degree of polynomial to be fitted when using Lasso.
#' 1 just fits the input X. 2 squares all variables and adds all pairwise
#' interactions. 3 squares and cubes all variables and adds all pairwise and
#' threewise interactions...
#' @param polynomial.Ridge degree of polynomial to be fitted when using Ridge,
#' see polynomial.Lasso for more info
#' @param polynomial.OLS degree of polynomial to be fitted when using OLS,
#' see polynomial.Lasso for more info
#' @param polynomial.Logit_lasso degree of polynomial to be fitted when using
#' Logit Lasso, see polynomial.Lasso for more info
#' @param polynomial.NLLS_exp degree of polynomial to be fitted when using
#' NLLS_exp, see polynomial.Lasso for more info
#' @param polynomial.loglin degree of polynomial to be fitted when using
#' loglin, see polynomial.Lasso for more info
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
#' @param ensemblefolds number of cross-fitting folds used in OLSensemble
#' @param start_nlls list with starting values for the NLLS_exp parameters.
#' Default uses log(mean(Y)) for intercept and zero for slopes.
#' @param intercept logical; should an intercept be included? Default TRUE.
#' Only applies to OLS, Lasso, Ridge, and loglin.
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
directMLate <- function(Y, X, D,
                        ML = c("Lasso", "Ridge", "RF", "CIF", "XGB", "CB",
                               "Torch", "grf", "SL", "OLSensemble"),
                        SL.library = c("SL.ranger"),
                        polynomial.Lasso = 1,
                        polynomial.Ridge = 1,
                        polynomial.OLS = 1,
                        polynomial.Logit_lasso = 1,
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
                        ensemblefolds = 10,
                        start_nlls = NULL,
                        intercept = TRUE) {

  if (!("data.frame" %in% class(X))) {
    X <- data.frame(X)
  }

  mu1 <- ML::modest(X[D == 1,], Y[D == 1], ML = ML,
                    SL.library = SL.library,
                    polynomial.Lasso = polynomial.Lasso,
                    polynomial.Ridge = polynomial.Ridge,
                    polynomial.OLS = polynomial.OLS,
                    polynomial.Logit_lasso = polynomial.Logit_lasso,
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
                    ensemblefolds = ensemblefolds,
                    start_nlls = start_nlls,
                    intercept = intercept)
  mu1 <- ML::FVest(mu1, X, Y, Xnew = X, Ynew = Y, ML = ML,
                   polynomial.Lasso = polynomial.Lasso,
                   polynomial.Ridge = polynomial.Ridge,
                   polynomial.OLS = polynomial.OLS,
                   polynomial.Logit_lasso = polynomial.Logit_lasso,
                   polynomial.NLLS_exp = polynomial.NLLS_exp,
                   polynomial.loglin = polynomial.loglin,
                   intercept = intercept)

  mu0 <- ML::modest(X[D == 0,], Y[D == 0], ML = ML,
                    SL.library = SL.library,
                    polynomial.Lasso = polynomial.Lasso,
                    polynomial.Ridge = polynomial.Ridge,
                    polynomial.OLS = polynomial.OLS,
                    polynomial.Logit_lasso = polynomial.Logit_lasso,
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
                    ensemblefolds = ensemblefolds,
                    start_nlls = start_nlls,
                    intercept = intercept)
  mu0 <- ML::FVest(mu0, X, Y, Xnew = X, Ynew = Y, ML = ML,
                   polynomial.Lasso = polynomial.Lasso,
                   polynomial.Ridge = polynomial.Ridge,
                   polynomial.OLS = polynomial.OLS,
                   polynomial.Logit_lasso = polynomial.Logit_lasso,
                   polynomial.NLLS_exp = polynomial.NLLS_exp,
                   polynomial.loglin = polynomial.loglin,
                   intercept = intercept)

  b3 <- round(mean(mu1 - mu0), 3)
  resdirect <- data.frame("ATE" = b3, "ML" = ML)
  return(resdirect)
}
