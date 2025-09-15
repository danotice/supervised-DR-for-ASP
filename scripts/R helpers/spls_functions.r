library(spls)
library(pls)
library(e1071)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)

relPerf <- function(perf_df, min_){
  Y_rel = perf_df %>%
    rowwise()
  
  if(min_){
    Y_rel = Y_rel %>%
      mutate(across(everything(), ~ {
        row_vals <- c_across(everything())
        min_val <- min(row_vals, na.rm = TRUE)
        if (min_val == 0) . else abs((. - min_val) / min_val)
      })) 
  }
  else{
    Y_rel = Y_rel %>%
      mutate(across(everything(), ~ {
        row_vals <- c_across(everything())
        max_val <- max(row_vals, na.rm = TRUE)
        if (max_val == 0) . else abs((. - max_val) / max_val)
      }))
  }
  
  Y_rel = Y_rel %>%
    ungroup() %>%
    mutate(across(everything(), ~ replace_na(., 0)))
  
  return(Y_rel)
}

project.spls <- function(object, newx){
  if(sum(object$meanx > 1e-6)){
    newx = sweep(newx, 2, object$meanx, FUN = "-")
  }
  if(sum(object$normx != 1)){
    newx = sweep(newx, 2, object$normx, FUN = "/")
  }
  
  subX = newx[names(newx)[object$A]]
  Z = as_tibble(as.matrix(subX) %*% object$projection)
  colnames(Z) = paste0("Z",1:ncol(object$projection))
  
  return(Z)
}

fit.splsClass <- function(x, y, best, eta, C, sigma_, scalers=c(FALSE,FALSE)){
  ## projection with best params
  best.spls = spls(x, y, scale.x = scalers[1], scale.y = scalers[2],
                   fit = "oscorespls", K=2, eta=eta)
  
  projS = project.spls(best.spls, x)
  projS = cbind(projS, Best=best)
  
  formulaP = as.formula("Best ~ Z1 + Z2")
  best.svm = svm(formulaP, data = projS, gamma = sigma_, cost = C, probability = TRUE)
  
  return(list(spls = best.spls, svm = best.svm))
}

proj.pred.spls <- function(spls.mod, svm.mod, x, best){
  projS = project.spls(spls.mod, x)
  predS = predict(svm.mod, projS, probability = TRUE)
  
  probs = rename_with(data.frame(attr(predS, "probabilities")), 
                      ~ paste0("prob_svm_", .))
  
  attr(predS, "probabilities") <- NULL
  attr(predS, "names") <- NULL
  
  return(cbind(projS, Best = best, pred_svm = predS, probs))
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

