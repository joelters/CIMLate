#' Estimate Average Treatment Effect (ATE) using a locally robust moment
#' and machine learning methods
#'
#' `LRate` estimates the ATE by using locally robust/orthogonal scores
#' which have the property of double robustness and for which inference
#' is available even when using machine learning if cross-fitting is employed
#'
#'
#' @param X is a dataframe containing all the covariates
#' @param Y is a vector containing the outcome
#' @param D the treatment indicator
#' @param MLreg is a string specifying which machine learner to use among
#' Lasso, Ridge, RF, CIF, XGB, CB, Torch, grf, OLSensemble or SuperLearner
#' to estimate response functions
#' @param MLps is a string specifying which machine learner to use among
#' RF, CIF, Logit Lasso, XGB, CB, grf or SuperLearner to estimate propensity scores
#' @param pscore vector with treatment probabilities if known (e.g. in an RCT).
#' Otherwise leave NULL so that they are estimated using MLps
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
#' @param CF whether cross-fitting should be used
#' @param K number of splits to use for cross-fitting
#' @returns dataframe with estimates and standard errors (if cross-fitting is employed)
#' @examples
#' n <- 1000
#' X <- rnorm(n)
#' D <- runif(n) >= 0.5
#' Y0 <- X + rnorm(n)
#' Y1 <- 5 + X + 5*X + rnorm(n)
#' Y <- Y1*D + Y0*(1-D)
#' LRate(Y,X,D, MLreg = "RF", MLps = "Logit_lasso")
#'
#' @export
LRate <- function(Y,
                  X,
                  D,
                  MLreg = c("Lasso", "Ridge", "RF", "CIF", "XGB", "CB",
                            "Torch", "grf", "SL", "OLSensemble"),
                  MLps = c("RF", "Logit_lasso", "CIF", "XGB", "CB", "grf", "SL"),
                  pscore = NULL,
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
                  intercept = TRUE,
                  CF = TRUE,
                  K = 2) {
  if (!("data.frame" %in% class(X))) {
    X <- data.frame(X)
  }

  # Helper to pass all tuning params to modest/MLest
  reg_args <- list(
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
    intercept = intercept
  )
  ps_args <- list(
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
    intercept = intercept
  )
  fv_reg_args <- list(
    polynomial.Lasso = polynomial.Lasso,
    polynomial.Ridge = polynomial.Ridge,
    polynomial.OLS = polynomial.OLS,
    polynomial.Logit_lasso = polynomial.Logit_lasso,
    polynomial.NLLS_exp = polynomial.NLLS_exp,
    polynomial.loglin = polynomial.loglin,
    intercept = intercept
  )

  if (CF == FALSE) {
    mu1 <- do.call(ML::modest, c(list(X = X[D == 1,], Y = Y[D == 1], ML = MLreg), reg_args))
    mu1 <- do.call(ML::FVest,  c(list(model = mu1, X = X, Y = Y, Xnew = X, Ynew = Y, ML = MLreg), fv_reg_args))
    mu0 <- do.call(ML::modest, c(list(X = X[D == 0,], Y = Y[D == 0], ML = MLreg), reg_args))
    mu0 <- do.call(ML::FVest,  c(list(model = mu0, X = X, Y = Y, Xnew = X, Ynew = Y, ML = MLreg), fv_reg_args))

    if (is.null(pscore) == TRUE) {
      ps <- do.call(ML::MLest, c(list(X = X, Y = D, ML = MLps), ps_args))
      if (sum(ps$FVs <= 0 | ps$FVs >= 1) > 0) {
        warning("There are estimated propensity scores <= 0 or >= 1,
                they have been changed to 0.001 or 0.999")
        ps$FVs <- ps$FVs * (ps$FVs > 0 | ps$FVs < 1) +
          0.001 * (ps$FVs <= 0) + 0.999 * (ps$FVs >= 1)
      }
    } else {
      ps <- list(FVs = pscore)
    }

    lr <- mu1 - mu0 + (D / ps$FVs) * (Y - mu1) - ((1 - D) / (1 - ps$FVs)) * (Y - mu0)
    b4 <- round(mean(lr), 3)
    reslr <- data.frame("ATE" = b4, "MLreg" = MLreg, "MLps" = MLps)
    return(reslr)
  } else {
    n <- length(Y)
    L <- K
    ind <- split(seq(n), seq(n) %% K)

    mu1 <- rep(0, n)
    mu0 <- rep(0, n)
    ps  <- rep(0, n)

    for (i in 1:L) {
      Xnoti <- X[-ind[[i]],]
      Dnoti <- D[-ind[[i]]]
      Ynoti <- Y[-ind[[i]]]

      if (!("data.frame" %in% class(Xnoti))) {
        Xnoti <- data.frame(Xnoti)
      }

      if (is.null(pscore) == TRUE) {
        mps <- do.call(ML::modest, c(list(X = Xnoti, Y = Dnoti, ML = MLps), ps_args))
      }
      mu1m <- do.call(ML::modest, c(list(X = Xnoti[Dnoti == 1,], Y = Ynoti[Dnoti == 1], ML = MLreg), reg_args))
      mu0m <- do.call(ML::modest, c(list(X = Xnoti[Dnoti == 0,], Y = Ynoti[Dnoti == 0], ML = MLreg), reg_args))

      Xi <- X[ind[[i]],]
      Di <- D[ind[[i]]]
      Yi <- Y[ind[[i]]]

      if (!("data.frame" %in% class(Xi))) {
        Xi <- data.frame(Xi)
      }

      mu1[ind[[i]]] <- do.call(ML::FVest, c(list(model = mu1m, X = Xnoti, Y = Ynoti, Xnew = Xi, Ynew = Yi, ML = MLreg), fv_reg_args))
      mu0[ind[[i]]] <- do.call(ML::FVest, c(list(model = mu0m, X = Xnoti, Y = Ynoti, Xnew = Xi, Ynew = Yi, ML = MLreg), fv_reg_args))

      if (is.null(pscore) == TRUE) {
        fv_ps_args <- list(
          polynomial.Lasso = polynomial.Lasso,
          polynomial.Ridge = polynomial.Ridge,
          polynomial.OLS = polynomial.OLS,
          polynomial.Logit_lasso = polynomial.Logit_lasso,
          polynomial.NLLS_exp = polynomial.NLLS_exp,
          polynomial.loglin = polynomial.loglin,
          intercept = intercept
        )
        ps[ind[[i]]] <- do.call(ML::FVest, c(list(model = mps, X = Xnoti, Y = Dnoti, Xnew = Xi, Ynew = Di, ML = MLps), fv_ps_args))
      } else {
        ps[ind[[i]]] <- pscore[ind[[i]]]
      }

      if (sum(ps[ind[[i]]] <= 0 | ps[ind[[i]]] >= 1) > 0) {
        warning("There are estimated propensity scores <= 0 or >= 1,
                they have been changed to 0.001 or 0.999")
        ps[ind[[i]]] <- ps[ind[[i]]] * (ps[ind[[i]]] > 0 | ps[ind[[i]]] < 1) +
          0.001 * (ps[ind[[i]]] <= 0) + 0.999 * (ps[ind[[i]]] >= 1)
      }
    }

    lr_cf <- mu1 - mu0 + (D / ps) * (Y - mu1) - ((1 - D) / (1 - ps)) * (Y - mu0)
    b5  <- mean(lr_cf)
    se5 <- sd(lr_cf) / sqrt(n)
    reslrcf <- data.frame("ATE" = b5, "se" = se5, "MLreg" = MLreg, "MLps" = MLps)
    return(reslrcf)
  }
}
