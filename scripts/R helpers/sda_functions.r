library(elasticnet)
library(MASS)

sda.run <- function(X,Y,q, tol, maxit, lambda, trace){
  #browser()
  N = nrow(Y)
  K = ncol(Y)
  
  Dpi <- diag(apply(Y,2,sum)/N)
  Qk = matrix(1,K,1)
  
  for(k in 1:q){
    ## (a)
    theta_s <- rnorm(K)
    projection <- Qk %*% (t(Qk) %*% Dpi)
    
    theta_k <- theta_s - projection %*% theta_s
    
    norm_theta <- sqrt(t(theta_k) %*% Dpi %*% theta_k)
    theta_k <- theta_k / as.numeric(norm_theta)
    
    for (iter in 1:maxit) {
      theta_old <- theta_k
      Ytheta = Y %*% theta_k / sqrt(N)
      
      ## (b.i) elasticnet to estimate beta
      beta <- elasticnet::solvebeta(X/sqrt(N), Ytheta, paras=c(lambda$l2, abs(lambda$l1)),sparse=lambda$sparse) 
      
      ## (b.ii)
      # Compute class scores: Dpi^{-1} * Y^T * X * beta
      theta_raw <- diag(1 / diag(Dpi)) %*% t(Y) %*% (X %*% beta)
      
      # Project onto orthogonal complement under Dpi-inner product
      projection <- Qk %*% (t(Qk) %*% Dpi)
      theta_k <- theta_raw - projection %*% theta_raw
      
      norm_theta <- sqrt(t(theta_k) %*% Dpi %*% theta_k)
      theta_k <- theta_k / as.numeric(norm_theta)
      
      # Check convergence
      #browser()
      conv_diff = sqrt(sum((theta_k - theta_old)^2))
      # if(trace) cat("it: ", iter, " diff: ", conv_diff, "\n")
      # if(trace & (iter==1)) cat("it: ", iter, round(diag(Dpi),4), "--", round(1 / diag(Dpi),2), "\n")
      if (conv_diff < tol) break
      
    }
    
    if(trace) cat("Dim ", k, " --- its ", iter, "\n")
    if (iter==maxit){warning('Forced exit. Maximum number of iterations reached in SDA step.')}
    
    ## (c)
    if(k<q){
      Qk <- cbind(Qk, theta_k)
    }
    
    if(k==1){
      Beta = matrix(beta, ncol = 1)
      Theta = matrix(theta_k,  ncol = 1)
    }else{
      Beta = cbind(Beta, beta)
      Theta = cbind(Theta, theta_k)
    }
    
  }
  return(list(Q = Qk, theta = Theta, beta = unname(Beta)))
}

fit.sda <- function(x,y,dim,nfeat,tol=1e-4,maxit=50, lambda2=1e-6, trace=FALSE){
  ## correct data structures
  if (!is.factor(y)) {
    y <- as.factor(y)
  }
  # 2. One-hot encode y → Y
  Y <- model.matrix(~ y - 1)
  colnames(Y) <- levels(y)
  
  if (is.data.frame(x)) {
    X <- as.matrix(x)
  }
  if (!is.numeric(X)) {
    stop("X must be a numeric matrix or data.frame with only numeric columns.")
  }
  
  # regularization params
  lambda = list(l1 = nfeat, l2= lambda2, sparse = ifelse(nfeat<0, "varnum", "penalty") )
  
  # 3. Call SDA
  fit = sda.run(X, Y, q=dim, tol=tol, maxit=maxit, lambda = lambda, trace = trace)
  
  # 4. sparsity
  notZero <- apply(fit$beta, 1, function(x) any(x != 0))
  keptVars = colnames(X)[notZero]
  # b <- as.matrix(fit$beta[notZero,])
  # X <- X[, notZero, drop = FALSE]
  # varNames <- colnames(X)
  
  Xbeta <- X %*% fit$beta
  
  # 6. Fit LDA in reduced space
  lda_fit <- MASS::lda(Xbeta, y)
  
  structure(list(
    beta = fit$beta,
    theta = fit$theta,
    nfeat = nfeat,
    origP = ncol(X),
    keptVars = keptVars,
    classes = levels(y),
    fit = lda_fit,
    sparseParams = lambda),
    class= "sda"
  )
}

# predict.sda <- function(object, newX){
#   if(class(object)!= "sda") stop("not an SDA model")
#   if(!is.matrix(newX)) newX <- as.matrix(newX)
#   if(ncol(newX) != object$origP) stop("dimensions of training and testing X different")
#     
#   Xb <- newX %*% object$beta
#   pred <- predict(object$fit,Xb)
#   
#   return(pred)
# }

init.subProb <- function(X,y,Rj, method){
  classKey <- rep(levels(y), times = Rj)
  Z <- matrix(0, nrow = nrow(X), ncol = sum(Rj))
  
  if(method=="kmeans"){
    tmp <- mda::mda.start(X, y, subclasses = Rj,  start.method = "kmeans")
    
    for(i in seq(along = tmp)){
      colIndex <- which(classKey == names(tmp)[i])
      rowIndex <- which(y == names(tmp)[i])
      Z[rowIndex, colIndex] <- tmp[[i]]
    }
    
  }else{
    
    tmp = rep(0,length(y))
    for(c in levels(y)){
      tmp[y==c] = sample(which(classKey==c), size=table(y)[c], replace = TRUE)
    }
    
    for(i in 1:length(classKey)){
      rowIndex = which(tmp==i)
      Z[rowIndex, i] = 1
    }
    
  }
  
  return(Z)
}

smda.ite <- function(X,Z,Y,Q, Rj,tol, maxit, lambda, trace){

  R = sum(Rj)
  N = nrow(X)
  classes = colnames(Y)
  classKey <- rep(classes, times = Rj)
  
  ## (a) solve with SDA
  sda.fit = sda.run(X,Z,q=Q,tol=tol, maxit=maxit, lambda = lambda, trace = trace)
  
  ## (b)
  Xtilde = X %*% sda.fit$beta
  
  ## (c) - Expectation ------
  mu = matrix(0,R,Q)
  dpi = rep(0,R)
  
  for (k in seq(along = classes)){
    
    rK <- which(classKey==classes[k]) ## subclass cols in class k
    iCk = which(Y[,k]==1)  # instances in class k
    
    ## eq 16
    dpi_kr <- apply(Z[iCk,rK, drop=FALSE],2,sum)  
    dpi[rK] <- dpi_kr/sum(dpi_kr)
    
    ## eq 17
    mu_kr = (t(Z[iCk,rK,drop = FALSE])) %*% Xtilde[iCk,,drop = FALSE]
    mu[rK,] = sweep(mu_kr, 1, dpi_kr, "/")
    
  }
  
  ## eq 18 ----
  Sigma = matrix(0,Q,Q)
  for (r in 1:R) {  ## -- Z has zeros so I don't need to filter here
    diff <- sweep(Xtilde, 2, mu[r, ], "-")  # (X_i β - μ_r) for all i
    weighted <- sqrt(Z[, r]) * diff         # apply sqrt(Z_ir) row-wise 
    Sigma <- Sigma + t(weighted) %*% weighted
  }
  Sigma = Sigma/N
  
  if (kappa(Sigma)>1e8){
    Sigma = Sigma + 1e-6*diag(rep(1,Q))
  }
  Sigma_inv <- solve(Sigma)
  
  ## (d-e) - Maximization -------
  Zp = matrix(0, nrow = N, ncol = R)
  ll = 0 
  
  for (k in seq(along = classes)){
    
    rK <- which(classKey==classes[k]) ## subclass cols in class k
    iCk = which(Y[,k]==1)  # instances in class k
    
    logZ_k = matrix(0, nrow = length(iCk), ncol=length(rK))
    for(r in seq(along = rK)){
      diff = sweep(Xtilde[iCk,,drop = FALSE], 2, mu[rK[r],], "-") # Xtilde - mu_kr
      quad <- rowSums((diff %*% Sigma_inv) * diff)  # Mahalanobis squared distance
      logZ_k[,r] = log(dpi[r]) - 0.5 * quad
    }
    # Normalize logZ row-wise to get Z using log-sum-exp trick
    logZk_max <- apply(logZ_k, 1, max, na.rm=TRUE)
    probs = exp(logZ_k - logZk_max)
    ll = ll + sum(logZk_max + log(rowSums(probs)))
    
    Zp[iCk, rK] = probs
    
    
  }
  
  Zp = Zp/rowSums(Zp)
  
    #browser()
    
  return(list(
    Z = Zp,
    beta = sda.fit$beta,
    theta = sda.fit$theta,
    mu = mu,
    Sigma = Sigma,
    Sigma_inv = Sigma_inv,
    dpi = dpi,
    logLik = ll
    
  ))
  
}


fit.smda <- function(x,y,Rj,Q,nfeat,tol=1e-4,maxit=50, itmult=1, lambda2=1e-6, trace=FALSE, repInit=1, initMethod="kmeans"){
  # regularization params
  lambda = list(l1 = nfeat, l2= lambda2, sparse = ifelse(nfeat<0, "varnum", "penalty") )
  
  ## correct data structures
  if (!is.factor(y)) y <- as.factor(y)
  Y <- model.matrix(~ y - 1)    # 2. One-hot encode y → Y
  if (is.data.frame(x)) X <- as.matrix(x)
  if (!is.numeric(X))  stop("X must be a numeric matrix or data.frame with only numeric columns.")
  
  N = nrow(X)
  classes = levels(y)
  K = length(classes)
  if(length(Rj) == 1) Rj <- rep(Rj, K)
  R = sum(Rj)
  
  subClasses <- c()
  for(i in 1:length(classes)){
    subClasses = c(subClasses, paste(classes[i], 1:Rj[i], sep = "."))
  }
  classKey <- rep(classes, times = Rj)
  
  fit.out = NULL
  conv = FALSE  # say if final model converged
  
  ## restart if not converged
  rerun = TRUE
  restarts = 0
  while(rerun){
    
    rss = c()
    fit.prev = NULL
    early.exit = FALSE
    
    ll_old = -Inf
    #RSSold = Inf
    
      
    ## (1-2) initial subclass probabilities
    Z = init.subProb(X,y,Rj,initMethod)
    if(trace) cat("Init ", restarts, " -- ", colSums(Z) , "\n")
    
    ## (3) - iterate til convergence
    for(iter in 1:maxit){
      fit = smda.ite(X,Z,Y,Q, Rj,tol, maxit*itmult, lambda, trace)
      
      #browser()
      ## break if any empty subclass 
      zSub = factor(subClasses[apply(fit$Z, 1, which.max)], levels = subClasses)
      if(min(table(zSub))==0){
        early.exit = TRUE
        break
      }
      
      ## Z is subclass probabilities
      Z = fit$Z
      colnames(Z) <- subClasses
      
      ## trying with Z indicator matrix
      # Z = model.matrix(~ zSub - 1)
      # colnames(Z) <- subClasses
      
      # Check convergence via logLik
      rss = c(rss, fit$logLik)
      if (trace) cat("EM Iteration:", iter, "  LogLik:", round(fit$logLik,2), "\n")
      
      if(abs((fit$logLik - ll_old)/fit$logLik) < tol) break
      ll_old = fit$logLik
      fit.prev = fit
      
      # Check convergence via RSS -- not changed for indicator Z
      # Ztheta <- Z%*%sda.fit$theta
      # RSS <- sum((Ztheta-Xtilde)^2)+lambda$l2*sum(sda.fit$beta^2)
      # rss = c(rss, RSS)
      
      # if (trace) cat("EM Iteration:", iter, "  RSS:", round(RSS,2), "\n")
      # if (abs(RSS - RSSold)/RSS < tol) break
      # RSSold = RSS
      
    }
    
    restarts = restarts + 1
    rerun = restarts < repInit
    
    if (early.exit){
      warning("STOPPING!! Empty subclasses present. Returning previous iteration.")
      
      if(is.null(fit.out)) {fit.out = fit.prev}
      else if (is.null(fit.prev)){}
      else if(fit.out$logLik < fit.prev$logLik) {fit.out = fit.prev}
      
    }
    else if (iter==maxit){
      warning('STOPPING! Maximum number of iterations reached in SMDA step.')
      
      if(is.null(fit.out)) {fit.out = fit}
      else if(fit.out$logLik < fit$logLik) {fit.out = fit}
      
    }else{
      fit.out = fit
      conv = TRUE
      rerun = FALSE
      ## check if stuck in local max
      #stuck = abs((max(rss) - fit$logLik)/max(rss)) > 1e-2
      #conv = !stuck
      #rerun = ifelse(conv, FALSE, rerun)
      
      if (trace) cat("CONVERGENCE\n")
    }
  
  }
  
  zSub = factor(subClasses[apply(fit.out$Z, 1, which.max)], levels = subClasses)
  
  # 4. sparsity
  notZero <- apply(fit.out$beta, 1, function(x) any(x != 0))
  keptVars = colnames(X)[notZero]
   
  Xbeta <- X %*% fit.out$beta
  
  # 6. Fit LDA in reduced space
  lda_fit <- MASS::lda(Xbeta, zSub)
  
  structure(list(
    Z = Z,
    Ysub = zSub,
    Rj = Rj,
    K = K,
    beta = fit.out$beta,
    theta = fit.out$theta,
    mu = fit.out$mu,
    Sigma = fit.out$Sigma,
    Sigma_inv = fit.out$Sigma_inv,
    dpi = fit.out$dpi,
    nfeat = nfeat,
    origP = ncol(X),
    keptVars = keptVars,
    classes = classes,
    subClasses = subClasses,
    logLik = rss,
    sparseParams = lambda,
    fit = lda_fit,
    conv = conv),
  class= "smda"
  )
}

predict.smda <- function(object, newX){
  if(class(object)!= "smda") stop("not an SMDA model")
  if(!is.matrix(newX)) newX <- as.matrix(newX)
  if(ncol(newX) != object$origP) stop("dimensions of training and testing X different")
  
  classes = object$classes
  classKey = rep(classes, times = object$Rj)
  
  Xb <- newX %*% object$beta
  subPred <- predict(object$fit,Xb)
  
  probs = matrix(0, nrow = nrow(newX), ncol = length(classes))
  for(k in seq(along = classes)){
    rK <- which(classKey==classes[k]) ## subclass cols in class k
    probs[,k] = rowSums(subPred$posterior[,rK, drop=FALSE])
    
  }
  class <- factor(classes[apply(probs,1,which.max)], levels = classes)
  colnames(probs) <- classes
  
  return(list(
    class=class,
    posterior = probs,
    subclass = subPred$class,
    subPosterior = subPred$posterior,
    x = Xb #subPred$x
    ))
}

print.smda <- function(object){
  str(object)
}

bic.smda <- function(object){
  loglik = object$logLik[length(object$logLik)]
  n.beta = sum(object$beta != 0)
  n.theta = prod(dim(object$theta))
  
  n = nrow(object$Z)
  
  return((n.beta + n.theta)*log(n) - 2*loglik)
}


# ####### Glass data ----------
# data("glass")
# glass$Type = factor(glass$Type)
# prep = preProcess(glass,c("center", "scale"))
# data_ = predict(prep, glass)
# 
# table(glass$Type)
# 
# sda.out = fit.sda(x = data_[,1:9], y = data_$Type, dim=2, nfeat=-4, maxit=400,trace = TRUE)
# 
# set.seed(121)
# smda.out = fit.smda(x = data_[,1:9], y = data_$Type, Rj=c(2,2,1,1,1,1), Q=2, nfeat=-3, maxit=400,trace = TRUE, repInit = 5)
# 
# 
# ####### PD data ----------
# # library(mlbench)
# data("PimaIndiansDiabetes")
# prep = preProcess(PimaIndiansDiabetes,c("center", "scale"))
# data_ = predict(prep, PimaIndiansDiabetes)
# 
# table(data_$diabetes)
# 
# smda.out = fit.smda(x = data_[,1:8], y = data_$diabetes, Rj=2, Q=2, nfeat=-3, maxit=400,trace = TRUE, repInit = 5)
# 
# 
# smda.out$dpi
# smda.out$rss
# View(smda.out$Z)
# colSums(smda.out$Z)
# 
# # Y = model.matrix(~ metadata_train$best - 1)
# # X = as.matrix(prep_train[feature_names])
