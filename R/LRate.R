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
#' Lasso, Ridge, Random Forests or XGBoosting to estimate response functions
#' @param MLps is a string specifying which machine learner to use among
#' Random Forests or Logit Lasso to estimate propensity scores
#' @param pscore vector with treatment probabilities if known (e.g. in an RCT).
#' Otherwise leave NULL so that they are estimated using MLps
#' @param polynomial degree of polynomial to be fitted when using Lasso, Ridge
#' or Logit Lasso. 1 just fits the input X. 2 squares all variables and adds
#' all pairwise interactions. 3 squares and cubes all variables and adds all
#' pairwise and threewise interactions...
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
                  MLreg = c("Lasso", "Ridge", "RF", "XGB","grf","SL"),
                  MLps = c("RF", "Logit_lasso","grf","SL"),
                  SL.library = c("SL.ranger"),
                  pscore = NULL,
                  polynomial = 1,
                  CF = TRUE,
                  K = 2){
  if (!("data.frame" %in% class(X))){
    X <- data.frame(X)
  }
  if (CF == FALSE){
    mu1 <- ML::modest(X[D == 1,], Y[D == 1], ML = MLreg, polynomial = polynomial,
                      SL.library = SL.library)
    mu1 <- ML::FVest(mu1,X,Y,Xnew = X, Ynew = Y, ML = MLreg, polynomial = polynomial)
    mu0 <- ML::modest(X[D == 0,], Y[D == 0], ML = MLreg, polynomial = polynomial,
                      SL.library = SL.library)
    mu0 <- ML::FVest(mu0,X,Y,Xnew = X, Ynew = Y, ML = MLreg, polynomial = polynomial)
    if (is.null(pscore) == TRUE){
      ps <- ML::MLest(X,D,ML = MLps, polynomial = polynomial,
                      SL.library = SL.library)
      if (sum(ps$FVs <= 0 | ps$FVs >= 1) > 0){
        warning("There are estimated propensity scores <= 0 or >= 1,
                they have been changed to 0.001 or 0.999")
        ps$FVs <- ps$FVs*(ps$FVs > 0 | ps$FVs < 1) +
          0.001*(ps$FVs <= 0) + 0.999*(ps$FVs >= 1)
      }
    }
    else{
      ps <- pscore
    }

    lr <- mu1 - mu0 + (D/ps$FVs)*(Y-mu1) - ((1-D)/(1-ps$FVs))*(Y-mu0)
    b4 <- round(mean(lr),3)
    # se4 <- sd(lr)/sqrt(length(Y))
    reslr <- data.frame("ATE" = b4,"MLreg" = MLreg, "MLps" = MLps)
    return(reslr)
  }
  else{
    n <- length(Y)
    L <- K
    ind <- split(seq(n), seq(n) %% K)

    mu1 <- rep(0,n)
    mu0 <- rep(0,n)
    ps <- rep(0,n)
    for (i in 1:L){
      Xnoti <- X[-ind[[i]],]
      Dnoti <- D[-ind[[i]]]
      Ynoti <- Y[-ind[[i]]]

      if (!("data.frame" %in% class(Xnoti))){
        Xnoti <- data.frame(Xnoti)
      }
      if (is.null(pscore) == TRUE){
        mps <- ML::modest(Xnoti[,], Dnoti, ML = MLps, polynomial = polynomial,
                          SL.library = SL.library)
      }
      mu1m <- ML::modest(Xnoti[Dnoti == 1,], Ynoti[Dnoti == 1], ML = MLreg, polynomial = polynomial,
                         SL.library = SL.library)
      mu0m <- ML::modest(Xnoti[Dnoti == 0,], Ynoti[Dnoti == 0], ML = MLreg, polynomial = polynomial,
                         SL.library = SL.library)

      Xi <- X[ind[[i]],]
      Di <- D[ind[[i]]]
      Yi <- Y[ind[[i]]]

      if (!("data.frame" %in% class(Xi))){
        Xi <- data.frame(Xi)
      }

      mu1[ind[[i]]] <- ML::FVest(mu1m, Xnoti, Ynoti, Xnew = Xi, Ynew = Yi, ML = MLreg, polynomial = polynomial)
      mu0[ind[[i]]] <- ML::FVest(mu0m, Xnoti, Ynoti, Xnew = Xi, Ynew = Yi, ML = MLreg, polynomial = polynomial)
      if (is.null(pscore) == TRUE){
        ps[ind[[i]]] <- ML::FVest(mps, Xnoti, Dnoti, Xnew = Xi, Ynew = Di, ML = MLps, polynomial)
      }
      else{
        ps[ind[[i]]] <- pscore[ind[[i]]]
      }

      if (sum(ps[ind[[i]]] <= 0 | ps[ind[[i]]] >= 1) > 0){
        warning("There are estimated propensity scores <= 0 or >= 1,
                they have been changed to 0.001 or 0.999")
        ps[ind[[i]]] <- ps[ind[[i]]]*(ps[ind[[i]]] > 0 | ps[ind[[i]]] < 1) +
          0.001*(ps[ind[[i]]] <= 0) + 0.999*(ps[ind[[i]]] >= 1)
      }

    }

    lr_cf <- mu1 - mu0 + (D/ps)*(Y-mu1) - ((1-D)/(1-ps))*(Y-mu0)
    b5 <- mean(lr_cf)
    se5 <- sd(lr_cf)/sqrt(n)
    reslrcf <- data.frame("ATE" = b5, "se" = se5,"MLreg" = MLreg, "MLps" = MLps)
    return(reslrcf)
  }
}
