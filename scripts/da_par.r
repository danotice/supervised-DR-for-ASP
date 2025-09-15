#!/usr/bin/env Rscript

library(tibble)
library(dplyr)
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

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

outfol <- args[1]
ncores <- as.integer(args[2])
#subFt <- logical(args[4])

######## functions ###############
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
A = length(levels(metadata_train$Best))

Z = c()
param.df = tibble(proj=c(), subclasses=c(),nFeat=c())

scaler = "s"
preAll = preProcess(metadata_train, 
                    method = case_when(scaler == "c" ~ c("center"),
                                       scaler == "s" ~ c("center","scale"),
                                       scaler == "p" ~ c("YeoJohnson"),
                                       .default =  c("center","scale"))
)
processed_train = predict(preAll, metadata_train)
processed_test = predict(preAll, metadata_test)


######### MAIN ###############

####### MDA proj ------
if(feat.rank >= length(selected_features)){
  
  mod.mda = rcv.mda.par(formula_, data = metadata_train, K=5, reps = 1,msub=10,d=2, 
                        scaler=scaler,grid.size=30, n.cores=ncores)
  best.mda = mod.mda$model
  
  name.mod = paste("MDA",scaler,sep="_")
  # if(sum(mod.mda$S)>A){ # add only if LDA not best -- not doing because I need to know if fitted
  Z.mda = unname(
    predict(best.mda, rbind(processed_train,processed_test), type = "var", dimension=2))
  
  Z.mda = tibble(
    proj = rep(name.mod,N),
    instances = c(processed_train$instances,processed_test$instances),
    group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test))),
    Z1 = Z.mda[,1], Z2 = Z.mda[,2],
    Best = c(processed_train[["Best"]],processed_test[["Best"]]),
    pred_da = predict(best.mda, rbind(processed_train,processed_test), type="class", dimension=2)
  )
  
  Z.mda = cbind(Z.mda,
                rename_with(data.frame(
                  predict(best.mda, rbind(processed_train,processed_test), type = "posterior", dimension=2)),
                  ~ paste0("prob_da_", .)))
  
  Z = rbind(Z, Z.mda)
  param.df = rbind(param.df, tibble(proj=name.mod,subclass=list(mod.mda$S),nFeat=NA))
  
  if(scaler=="s"){
    mod.mda.s = mod.mda
    best.mda.s = best.mda
  }else{
    mod.mda.c = mod.mda
    best.mda.c = best.mda
  }
  rm(mod.mda,best.mda)
  
}#}

#### fit LDA #####
best.lda = lda(formula_, data = processed_train)
lda.pr = predict(best.lda, rbind(processed_train,processed_test), dimen = 2)
name.mod = paste("LDA",scaler,sep="_")

Z.lda = tibble(
  proj = rep(name.mod,N),
  instances = c(processed_train$instances,processed_test$instances),
  group = c(rep("train",nrow(processed_train)), rep("test",nrow(processed_test))),
  Z1 = unname(lda.pr$x[,1]),
  Z2 = ifelse(rep(dim(best.lda$scaling)[2]==1,N), 
              NA_real_, unname(lda.pr$x[,2])),
  Best = c(processed_train[["Best"]],processed_test[["Best"]]),
  pred_da = lda.pr$class
)

Z.lda = cbind(Z.lda,
              rename_with(data.frame(lda.pr$posterior), ~ paste0("prob_da_", .)))

Z = rbind(Z, Z.lda)

if(scaler=="s"){
  best.lda.s = best.lda
}else{
  best.lda.c = best.lda
}
rm(best.lda)


#### fit PLSDA ####

best.plsda = plsda(x = rbind(metadata_train[selected_features],metadata_test[selected_features]), 
                   y = c(metadata_train$Best,metadata_test$Best), 
                   subset=1:length(metadata_train),
                   method="oscorespls",
                   ncomp = 2,validation="none",
                   scale = TRUE, center = TRUE)

## may or may not be broken!
attr(best.plsda,"class") = c("mvr", "plsda")
plsda.x = predict(best.plsda,rbind(metadata_train[selected_features],metadata_test[selected_features]),type = "scores")

Z.plsda = tibble(
  proj = rep("PLSDA",N),
  instances = c(metadata_train$instances,metadata_test$instances),
  group = c(rep("train",nrow(metadata_train)), rep("test",nrow(metadata_test))),
  Z1 = plsda.x[,1],
  Z2 = plsda.x[,2],
  Best = c(metadata_train[["Best"]],metadata_test[["Best"]])
)

attr(best.plsda,"class") = c("plsda","mvr")
Z.plsda = mutate(Z.plsda,
                 pred_da = predict(best.plsda, rbind(metadata_train[selected_features],metadata_test[selected_features]), type = 'class')
)
Z.plsda = cbind(Z.plsda,
                rename_with(data.frame(
                  predict(best.plsda,rbind(metadata_train[selected_features],metadata_test[selected_features]), 
                          type="prob", probMethod = "Bayes")[,,1]), ~ paste0("prob_da_", .)))


Z = rbind(Z, Z.plsda)


############ OUTPUT ###########

### write to csv
Z %>%
  mutate(across(starts_with("prob_"), ~ round(.x, 5))) %>% 
  write_csv(paste0(outfol,"da_proj.csv"))

if(nrow(param.df) > 0){
  param.df %>% 
    mutate(subclass = sapply(subclass, toString)) %>%
    write_csv(paste0(outfol,"da_params.csv"))
}

### save objects
save(list = ls(pattern = "^(best\\.|mod\\.)"), file = paste0(outfol,"da_proj.Rdata"))

