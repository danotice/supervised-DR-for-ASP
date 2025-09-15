library(spls)
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