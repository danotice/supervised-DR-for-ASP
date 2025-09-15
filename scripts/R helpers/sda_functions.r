library(elasticnet)
library(MASS)
library(mda)
library(sparseLDA)


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


rcv.mda.par <- function(formula, data, K, reps, msub, d, scaler, grid.size=0, n.cores) {
  set.seed(1111)
  
  ylab = all.vars(formula)[1]
  nclass = length(levels(data[[ylab]]))
  
  # subclasses to tune
  s_list <- setNames(rep(list(1:msub), nclass), paste0(1:nclass))
  subclasses <- do.call(expand.grid, s_list)
  subclasses <- map(pmap(subclasses, c),unname)[-1]
  
  if (grid.size > 0 & grid.size < length(subclasses)){
    subclasses = unique(c(sample(subclasses, grid.size),
                          lapply(2:msub, function(x) rep(x, nclass))))
  }
  
  # Results storage
  Accuracy = matrix(nrow=length(subclasses),ncol=K*reps)
  
  for (r in 1:reps) {
    print(r)
    
    folds = createFolds(data[[ylab]], K)
    
    for (k in 1:K) {
      # Split into training and testing
      train_data <- data[-folds[[k]], ]
      test_data <- data[folds[[k]], ]
      
      pre = preProcess(train_data, 
                       method = case_when(scaler == "c" ~ c("center"),
                                          scaler == "s" ~ c("center","scale"),
                                          scaler == "p" ~ c("YeoJohnson"),
                                          .default =  c("center","scale"))
      )
      
      train_data = predict(pre, train_data)
      test_data = predict(pre, test_data)
      
      # defined after data so it will work
      run.subclass = function(s){
        # print(s)
        
        # Train model
        model <- tryCatch(
          mda(formula, data = train_data, subclasses = s, dimension = d,iter=10, tries=10),
          error = function(e){ 
            warning(paste("Unable to fit model: fold",k," rep",r,
                          " subclasses", paste(s, collapse=",")))
            return(NULL)  # Return NULL if an error occurs
          }
        )
        
        # If model is NULL, accuracy stays NA, otherwise proceed with predictions
        if (!is.null(model)) {
          predictions <- predict(model, test_data, type="class")
          # Calculate accuracy
          acc <- confusionMatrix(predictions, data[[ylab]][folds[[k]]])$overall['Accuracy']
        }else{
          acc = NA_real_
        }
        return(unname(acc))
      }
      # Loop over subclass combinations -- PARALLEL
      Accuracy[,K*(r-1)+k] = mcmapply(FUN=run.subclass, s=subclasses,mc.cores = n.cores)
      
    }
    # print(results)
  }
  
  AccuracyM = rowMeans(Accuracy, na.rm = TRUE)
  Accuracy = tibble(Accuracy)
  Accuracy$subclass = subclasses
  
  bestS = subclasses[[which.max(AccuracyM)]]
  preAll = preProcess(data, 
                      method = case_when(scaler == "c" ~ c("center"),
                                         scaler == "s" ~ c("center","scale"),
                                         scaler == "p" ~ c("YeoJohnson"),
                                         .default =  c("center","scale"))
  )
  bestModel = mda(formula, predict(preAll,data), subclasses = bestS, dimension = d,iter=10, tries=10)
  
  print(bestS)
  print(max(AccuracyM))
  
  
  return(list(model=bestModel, S=bestS, accuracy=Accuracy))
  
}

rcv.slda.par <- function(formula, data, K, reps, nvar.list, d, scaler, n.cores){
  cl = makeForkCluster(n.cores)
  registerDoParallel(cl)
  
  train_control <- trainControl(
    method = "repeatedcv", 
    number = K, repeats = reps
  )
  tune_grid <- expand.grid(NumVars = nvar.list, lambda = 1e-6) 
  
  sel.slda = train(
    formula, data = data,
    method = "sparseLDA",
    trControl = train_control,
    metric = "Accuracy",
    tuneGrid = tune_grid,
    preProcess = case_when(scaler == "c" ~ c("center"),
                           scaler == "s" ~ c("center","scale"),
                           scaler == "p" ~ c("YeoJohnson"),
                           .default =  c("center","scale")),
    Q = d, maxIte = 5000, tol = 1e-4
  )
  
  stopCluster(cl)
  print(sel.slda$results[sel.slda$results$NumVars == sel.slda$bestTune$NumVars, c("NumVars","Accuracy")])
  
  return(list(model=sel.slda$finalModel, params=list(NumVars=sel.slda$bestTune$NumVars), accuracy=sel.slda$results))
}

sel.smda.par <- function(formula, data, nvar.list, msub, d, grid.size=0, n.cores){
  ## nvar.list should have negative values
  ## takes scaled data
  
  set.seed(1111)
  
  ylab = all.vars(formula)[1]
  features = all.vars(formula)[2:length(all.vars(formula))]
  nclass = length(levels(data[[ylab]]))
  
  # subclasses to tune
  s_list <- setNames(rep(list(1:msub), nclass), paste0(1:nclass))
  subclasses <- do.call(expand.grid, s_list)
  subclasses <- map(pmap(subclasses, c),unname)[-1]
  #subclasses <- map(pmap(subclasses, c),unname)[-1]
  
  param.grid <- expand.grid(
    s_idx = seq_along(subclasses),
    nv = nvar.list,
    KEEP.OUT.ATTRS = FALSE
  )
  param.grid$s <- lapply(param.grid$s_idx, function(i) subclasses[[i]])
  
  if (grid.size > 0 & grid.size < nrow(param.grid)){
    param.grid$subConst <- sapply(param.grid$s, function(v) length(unique(v))==1)
    
    if(sum(param.grid$subConst) >= grid.size){
      param.grid = param.grid[param.grid$subConst, ]
    }else{
      grid.varied <- param.grid[!param.grid$subConst, ]
      grid.varied <- grid.varied[sample(nrow(grid.varied), grid.size-sum(param.grid$subConst)), ]
      
      param.grid = rbind(
        param.grid[param.grid$subConst, ],
        grid.varied)
      
    }
    
  }
  print(paste0("Checking parameter grid ", nrow(param.grid)))
  
  run.params <- function(s, nv){
    smda.mod = tryCatch(
      fit.smda(x=data[features], y=data[[ylab]], Rj=s, Q=d, nfeat=nv,
               maxit=300, itmult=5,repInit=3, tol=1e-3),
      error = function(e){
        warning(paste("Unable to fit model: nvar",nv,
                      " subclasses", paste(s, collapse=",")))
        return(NULL)  # Return NULL if an error occurs
      }
    )
    
    if(is.null(smda.mod)){
      return(list(model=NULL, bic=Inf, conv=FALSE))
    }else if(is.null(smda.mod$logLik)){
      return(list(model=smda.mod, bic=NA_real_, conv=smda.mod$conv))
    }else{
      return(list(model=smda.mod, bic=bic.smda(smda.mod), conv=smda.mod$conv))
    }
  }
  
  
  #models = mcmapply(run.params, s=param.grid$s, nv=param.grid$nv, mc.cores = n.cores,SIMPLIFY = FALSE)
  models = pbmcmapply(run.params, s=param.grid$s, nv=param.grid$nv, mc.cores = n.cores,SIMPLIFY = FALSE,
                      ignore.interactive=TRUE)
  
  bics <- sapply(models, function(m) if (isTRUE(m$conv)) m$bic else NA)
  best.mod = models[[which.min(bics)]]
  
  param.grid = cbind(
    param.grid[c('nv','s')],
    bic = bics,
    conv = sapply(models, function(m) m$conv))
  
  cat("subclasses ", best.mod$model$Rj, " nfeat ", best.mod$model$nfeat)
  
  return(list(scores = param.grid,
              model = best.mod$model))
  
}

rcv.smda.par <- function(formula, data, K, reps, msub, nvar.list, d, scaler, grid.size=0, n.cores, initMethod='kmeans'){
  ## nvar.list should have negative values
  ## takes scaled data
  
  set.seed(1111)
  
  ylab = all.vars(formula)[1]
  features = all.vars(formula)[2:length(all.vars(formula))]
  nclass = length(levels(data[[ylab]]))
  #browser()
  
  # subclasses to tune
  s_list <- setNames(rep(list(1:msub), nclass), paste0(1:nclass))
  subclasses <- do.call(expand.grid, s_list)
  subclasses <- map(pmap(subclasses, c),unname)[-1]
  
  param.grid <- expand.grid(
    s_idx = seq_along(subclasses),
    nv = nvar.list,
    KEEP.OUT.ATTRS = FALSE
  )
  param.grid$s <- lapply(param.grid$s_idx, function(i) subclasses[[i]])
  
  if (grid.size > 0 & grid.size < nrow(param.grid)){
    param.grid$subConst <- sapply(param.grid$s, function(v) length(unique(v))==1)
    
    if(sum(param.grid$subConst) >= grid.size){
      param.grid = param.grid[param.grid$subConst, ]
    }else{
      grid.varied <- param.grid[!param.grid$subConst, ]
      grid.varied <- grid.varied[sample(nrow(grid.varied), grid.size-sum(param.grid$subConst)), ]
      
      param.grid = rbind(
        param.grid[param.grid$subConst, ],
        grid.varied)
      
    }
    
  }
  print(paste0("Checking parameter grid ", nrow(param.grid)))
  
  # Results storage
  Accuracy = matrix(nrow=nrow(param.grid),ncol=K*reps)
  
  #folds = create_folds(data[[ylab]], K, m_rep = reps, use_names = FALSE, invert = TRUE)
  
  for (r in 1:reps) {
    print(r)
    
    folds = createFolds(data[[ylab]], K) 
    
    for (k in 1:K) {
      
      #f = (r-1)*K + k 
      # Split into training and testing
      train_data <- data[-folds[[k]], ]
      test_data <- data[folds[[k]], ]
      
      pre = preProcess(train_data, 
                       method = case_when(scaler == "c" ~ c("center"),
                                          scaler == "s" ~ c("center","scale"),
                                          scaler == "p" ~ c("YeoJohnson"),
                                          .default =  c("center","scale"))
      )
      
      train_data = predict(pre, train_data)
      test_data = predict(pre, test_data)
      
      run.params <- function(s, nv){
        smda.mod = tryCatch(
          fit.smda(x=train_data[features], y=train_data[[ylab]], Rj=s, Q=d, nfeat=nv,
                   maxit=300, itmult=5,repInit=5, tol=1e-3, initMethod = initMethod),
          error = function(e){
            warning(paste("Unable to fit model: nvar",nv,
                          " subclasses", paste(s, collapse=",")))
            return(NULL)  # Return NULL if an error occurs
          }
        )
        
        # If model is NULL, accuracy stays NA, otherwise proceed with predictions
        if (!is.null(smda.mod)) {
          #if (smda.mod$conv){
          predictions <- predict.smda(smda.mod, test_data[features])$class
          # Calculate accuracy
          acc <- confusionMatrix(predictions, test_data[[ylab]])$overall['Accuracy']
          # }else{
          #   acc = NA_real_
          # }
        }else{
          acc = NA_real_
        }
        return(unname(acc))
      }
      
      # Loop over subclass combinations -- PARALLEL
      out = mcmapply(run.params, s=param.grid$s, nv=param.grid$nv, mc.cores = n.cores)
      #browser()
      Accuracy[,(r-1)*K + k] = out
      
    }
  }
  
  AccuracyM = rowSums(Accuracy, na.rm = TRUE)/ncol(Accuracy)
  Accuracy = as_tibble(Accuracy)
  Accuracy$subclass = param.grid$s
  Accuracy$nfeat = param.grid$nv
  
  bestS = list(subclass=param.grid$s[[which.max(AccuracyM)]],
               nfeat=param.grid$nv[[which.max(AccuracyM)]])
  preAll = preProcess(data, 
                      method = case_when(scaler == "c" ~ c("center"),
                                         scaler == "s" ~ c("center","scale"),
                                         scaler == "p" ~ c("YeoJohnson"),
                                         .default =  c("center","scale"))
  )
  #browser()
  bestModel = fit.smda(x=predict(preAll,data)[features], y=data[[ylab]], Rj=bestS$subclass, Q=d, nfeat=bestS$nfeat,
                       maxit=300, itmult=5,repInit=5, tol=1e-3)
  
  print(bestS)
  print(max(AccuracyM))
  
  return(list(model=bestModel, S=bestS, accuracy=Accuracy))
}
