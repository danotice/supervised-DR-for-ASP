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

source("./R helpers/sda_functions.r")

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

## defaults 
cv = FALSE
cv.grid = 30

if (length(args)>=2){
  outfol <- args[1]
  ncores <- as.integer(args[2])
}else if (length(args)==3){
  cv = TRUE
  if (!is.na(as.integer(args[3]))){
    cv.grid = as.integer(args[3])
  }
}else{
  stop('invalid number of arguments. must be 2 or 3 (if doing cv).', call. = FALSE)
}

#subFt <- logical(args[4])

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
  
  if (cv){
    mod.mda = rcv.mda.par(formula_, data = metadata_train, K=5, reps = 1,msub=10,d=2, 
                          scaler=scaler,grid.size=30, n.cores=ncores)
    best.mda = mod.mda$model
  } else {
    mod.mda = NULL
    best.mda = mda(formula_, processed_train, subclasses = 3, dimension = 2,iter=10, tries=10)
  }
  
  
  
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
  param.df = rbind(param.df, tibble(proj=name.mod,subclass=
                                      ifelse(cv,list(mod.mda$S),3),nFeat=NA))
  
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


# #### fit PLSDA ####
# 
# best.plsda = plsda(x = rbind(metadata_train[selected_features],metadata_test[selected_features]), 
#                    y = c(metadata_train$Best,metadata_test$Best), 
#                    subset=1:length(metadata_train),
#                    method="oscorespls",
#                    ncomp = 2,validation="none",
#                    scale = TRUE, center = TRUE)
# 
# ## may or may not be broken!
# attr(best.plsda,"class") = c("mvr", "plsda")
# plsda.x = predict(best.plsda,rbind(metadata_train[selected_features],metadata_test[selected_features]),type = "scores")
# 
# Z.plsda = tibble(
#   proj = rep("PLSDA",N),
#   instances = c(metadata_train$instances,metadata_test$instances),
#   group = c(rep("train",nrow(metadata_train)), rep("test",nrow(metadata_test))),
#   Z1 = plsda.x[,1],
#   Z2 = plsda.x[,2],
#   Best = c(metadata_train[["Best"]],metadata_test[["Best"]])
# )
# 
# attr(best.plsda,"class") = c("plsda","mvr")
# Z.plsda = mutate(Z.plsda,
#                  pred_da = predict(best.plsda, rbind(metadata_train[selected_features],metadata_test[selected_features]), type = 'class')
# )
# Z.plsda = cbind(Z.plsda,
#                 rename_with(data.frame(
#                   predict(best.plsda,rbind(metadata_train[selected_features],metadata_test[selected_features]), 
#                           type="prob", probMethod = "Bayes")[,,1]), ~ paste0("prob_da_", .)))
# 
# 
# Z = rbind(Z, Z.plsda)


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

## update status if successful
file.create(paste0(outfol,"status.txt"))
fileConn<-file(paste0(outfol,"status.txt"))
writeLines("1", fileConn)
close(fileConn)