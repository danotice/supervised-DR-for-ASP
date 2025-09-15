#!/usr/bin/env Rscript

library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(parallel)
library(doParallel)
library(readr)
library(mda)
library(pls)
library(sparseLDA)
library(MASS)
library(caret)
#library(splitTools)
library(Matrix)
library(spls)
library(e1071)
library(pbmcapply)

source("./R helpers/sda_functions.r")
source("./R helpers/spls_functions.r")

# outfol <- '../Results/tsp0_all/'
# ncores=3
# scaler = "s"

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

outfol <- args[1]
ncores <- as.integer(args[2])

scaler = "s"
#ftSeq = 1:16
etaSeq = 0.02

######## functions ###############
# need to do the check for 1D projection outside of function

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

rcv.spls.par <- function(x, y, best, K, reps, eta.int, scalers=c(FALSE,FALSE), n.cores){
  cl = makeForkCluster(n.cores)
  registerDoParallel(cl)
  
  As = c(Inf)
  eta.list = seq(0,0.99,eta.int)
  
  pred.accuracySVM = tibble() 
  
  formulaP = as.formula("Best ~ Z1 + Z2")
  train_control <- trainControl(
    method = "repeatedcv", 
    number = K, repeats = reps
  )
  tune_gridRad <- expand.grid(C = 10^(-3:3), sigma =10^(-3:3))
  
  for(i in 1:length(eta.list)){
    
    print(eta.list[i])
    mod = spls(x, y, scale.x = scalers[1], scale.y = scalers[2],
               fit = "oscorespls",
               K=2, eta=eta.list[i])
    
    if(length(mod$A) != As[length(As)]){
      projS = project.spls(mod,x)
      projS = cbind(projS, Best= best)
      
      set.seed(111)
      sel.svmR = train(
        formulaP, data = projS,
        method = "svmRadial",
        trControl = train_control,
        metric = "Accuracy",
        tuneGrid = tune_gridRad,
        preProcess = c("center","scale")
      )
      
      pred.accuracySVM = rbind(
        pred.accuracySVM,
        tibble(eta = eta.list[i], feats = length(mod$A), C = sel.svmR$bestTune$C, sigma = sel.svmR$bestTune$sigma, accuracy = max(sel.svmR$results$Accuracy))
      )
    }
    
    As = c(As, length(mod$A))
  }
  
  best.params = pred.accuracySVM[which.max(pred.accuracySVM$accuracy),]
  
  stopCluster(cl)
  print(best.params)
  
  return(best.params)
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

###### DATA ###############

metadata_train <- read_csv(paste0(outfol,"metadata_train.csv")) 
metadata_train$Best = as.factor(metadata_train$Best)

metadata_test <- read_csv(paste0(outfol,"metadata_test.csv"))
metadata_test$Best = as.factor(metadata_test$Best)

feature_names = grep("^feature_", names(metadata_train), value = TRUE)
selected_features <- read_csv(paste0(outfol,"selected_features.csv"))
selected_features = unlist(selected_features[1,], use.names = FALSE)[-1]

formula_ = as.formula(paste("Best ~", paste(selected_features, collapse = " + ")))

feats = as.matrix(metadata_train[selected_features])
feat.rank = rankMatrix(feats)[1]
N = nrow(metadata_train)+nrow(metadata_test)

algs = names(metadata_train)[startsWith(names(metadata_train), "algo_")]
A = length(algs)

preAll = preProcess(metadata_train, 
                    method = case_when(scaler == "c" ~ c("center"),
                                       scaler == "s" ~ c("center","scale"),
                                       scaler == "p" ~ c("YeoJohnson"),
                                       .default =  c("center","scale"))
)
processed_train = predict(preAll, metadata_train)
processed_test = predict(preAll, metadata_test)

relperf_train = relPerf(metadata_train[algs], TRUE)
relperf_test = relPerf(metadata_test[algs], TRUE)

Z = c()
Z.svm = c()

param.df = tibble(proj=c(), subclasses=c(), nFeat=c(), eta=c())

######### MAIN -- Sparse methods ###############

##### SPLS + SVM ---------
best.spls = rcv.spls.par(x=processed_train[selected_features], y=processed_train[algs], best=processed_train$Best,
                         K=5, reps = 2, eta.int = etaSeq, n.cores=ncores)

mod.spls = fit.splsClass(x=processed_train[selected_features], y=processed_train[algs], best=processed_train$Best,
                         eta=best.spls$eta, C=best.spls$C, sigma_ = best.spls$sigma)

Z.spls = rbind(
  proj.pred.spls(mod.spls$spls, mod.spls$svm, processed_train[selected_features], processed_train$Best),
  proj.pred.spls(mod.spls$spls, mod.spls$svm, processed_test[selected_features], processed_test$Best)
)

Z.spls = cbind(
  proj = rep("SPLS",N),
  instances = c(processed_train$instances,processed_test$instances),
  group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test))),
  Z.spls)

Z.svm = rbind(Z.svm, Z.spls)
param.df = rbind(param.df, tibble(proj="SPLS",subclass=NA,nFeat=best.spls$feats, eta=best.spls$eta))

##### rel SPLS + SVM ---------
best.srpls = rcv.spls.par(x=processed_train[selected_features], y=relperf_train, best=processed_train$Best,
                          K=5, reps = 2, eta.int = etaSeq, scalers = c(FALSE,TRUE), n.cores=ncores)

mod.srpls = fit.splsClass(x=processed_train[selected_features], y=relperf_train, best=processed_train$Best,
                          eta=best.srpls$eta, C=best.srpls$C, sigma_ = best.srpls$sigma, scalers = c(FALSE,TRUE))

Z.srpls = rbind(
  proj.pred.spls(mod.srpls$spls, mod.srpls$svm, processed_train[selected_features], processed_train$Best),
  proj.pred.spls(mod.srpls$spls, mod.srpls$svm, processed_test[selected_features], processed_test$Best)
)

Z.srpls = cbind(
  proj = rep("SrPLS",N),
  instances = c(processed_train$instances,processed_test$instances),
  group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test))),
  Z.srpls)

Z.svm = rbind(Z.svm, Z.srpls)
param.df = rbind(param.df, tibble(proj="SrPLS",subclass=NA, nFeat=best.srpls$feats, eta=best.srpls$eta))


##### SDA ------------
mod.slda = rcv.slda.par(formula_, data = metadata_train, K=5, reps = 2,
                        nvar.list = 1:min(50, feat.rank, length(feature_names)),
                        d=min(2,A-1),
                        scaler=scaler, n.cores=ncores)
best.slda = mod.slda$model

slda.pr = sparseLDA::predict.sda(best.slda, rbind(processed_train,processed_test)[feature_names], dimen = 2)

Z.slda = tibble(
  proj = rep("SLDA",N),
  instances = c(processed_train$instances,processed_test$instances),
  group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test))),
  Z1 = unname(slda.pr$x[,1]),
  Z2 = ifelse(rep(dim(best.slda$fit$scaling)[2]==1,N),
              NA_real_, unname(slda.pr$x[,2])),
  Best = c(processed_train$Best,processed_test$Best),
  pred_da = slda.pr$class
)

Z.slda = cbind(Z.slda,
               rename_with(data.frame(slda.pr$posterior), ~ paste0("prob_da_", .)))

Z = rbind(Z, Z.slda)
param.df = rbind(param.df, tibble(proj="SLDA",subclass=NA,nFeat=mod.slda$params$NumVars, eta=NA))

##### SMDA -----------

# mod.smda = sel.smda.par(formula_, processed_train, nvar.list=-1:-min(50, feat.rank, length(feature_names)),
#                         msub=8, d=2,grid.size=500, n.cores=ncores)
# 
# best.smda = mod.smda$model
# smda.pr = predict.smda(best.smda, rbind(processed_train,processed_test)[feature_names])
# colnames(smda.pr$x) <- paste0("Z",seq_len(ncol(smda.pr$x)))
# 
# Z.smda = tibble(
#   proj = rep("SMDA",N),
#   instances = c(processed_train$instances,processed_test$instances),
#   group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test)))
# )
# 
# Z.smda = cbind(
#   Z.smda,
#   smda.pr$x,
#   Best = c(processed_train$Best,processed_test$Best),
#   pred_da = smda.pr$class,
#   rename_with(data.frame(smda.pr$posterior), ~ paste0("prob_da_", .))
# )
# 
# Z = rbind(Z, Z.smda)
# param.df = rbind(param.df, tibble(proj="SMDA",subclass=list(best.smda$Rj),nFeat=best.smda$nfeat, eta=NA))

##### SMDA - cv -------

mod.smda.cv = rcv.smda.par(formula_, data = metadata_train, K=5, reps = 2,
                        msub=8,
                        nvar.list = -1:-min(50, feat.rank, length(feature_names)),
                        d=2, grid.size = 1000,
                        scaler=scaler, n.cores=ncores)
best.smda.cv = mod.smda.cv$model

smda.pr = predict.smda(best.smda.cv, rbind(processed_train,processed_test)[feature_names])
colnames(smda.pr$x) <- paste0("Z",seq_len(ncol(smda.pr$x)))

Z.smda = tibble(
  proj = rep("SMDA",N),
  instances = c(processed_train$instances,processed_test$instances),
  group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test)))
)

Z.smda = cbind(
  Z.smda,
  smda.pr$x,
  Best = c(processed_train$Best,processed_test$Best),
  pred_da = smda.pr$class,
  rename_with(data.frame(smda.pr$posterior), ~ paste0("prob_da_", .))
)

param.df = rbind(param.df, tibble(proj="SMDA",subclass=list(best.smda.cv$Rj),nFeat=best.smda.cv$nfeat, eta=NA))

###### write to csv -----
Z %>%
  mutate(across(starts_with("prob_"), ~ round(.x, 5))) %>%
  write_csv(paste0(outfol,"sparse/sda_proj.csv"))

Z.svm %>%
  mutate(across(starts_with("prob_"), ~ round(.x, 5))) %>%
  write_csv(paste0(outfol,"sparse/spls_proj.csv"))

param.df %>%
  mutate(subclass = sapply(subclass, toString)) %>%
  write_csv(paste0(outfol,"sparse/sparse_params.csv"))

### save objects
save(list = ls(pattern = "^(best\\.|mod\\.)"), file = paste0(outfol,"sparse/sparse_proj.Rdata"))
