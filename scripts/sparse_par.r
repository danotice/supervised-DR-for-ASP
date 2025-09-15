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

## defaults 
cv = FALSE
cv.grid = 30
scaler = "s"
etaSeq = 0.02
sda_ft = 6
spls_defaults = list(eta=0.9, C=1, sigma=1)

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
if (cv){
  best.spls = rcv.spls.par(x=processed_train[selected_features], y=processed_train[algs], best=processed_train$Best,
                           K=5, reps = 2, eta.int = etaSeq, n.cores=ncores)
} else {
  best.spls = spls_defaults
}

mod.spls = fit.splsClass(x=processed_train[selected_features], y=processed_train[algs], best=processed_train$Best,
                         eta=best.spls$eta, C=best.spls$C, sigma_ = best.spls$sigma)
if (!cv){best.spls$feats = length(mod.spls$A)}

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
if (cv){
  best.srpls = rcv.spls.par(x=processed_train[selected_features], y=relperf_train, best=processed_train$Best,
                          K=5, reps = 2, eta.int = etaSeq, scalers = c(FALSE,TRUE), n.cores=ncores)
} else {
  best.srpls = spls_defaults
}

mod.srpls = fit.splsClass(x=processed_train[selected_features], y=relperf_train, best=processed_train$Best,
                          eta=best.srpls$eta, C=best.srpls$C, sigma_ = best.srpls$sigma, scalers = c(FALSE,TRUE))
if (!cv){best.srpls$feats = length(mod.srpls$A)}

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
if (cv) {
  mod.slda = rcv.slda.par(formula_, data = metadata_train, K=5, reps = 2,
                        nvar.list = 1:min(50, feat.rank, length(feature_names)),
                        d=min(2,A-1),
                        scaler=scaler, n.cores=ncores)
  best.slda = mod.slda$model
} else {
  mod.slda = NULL
  best.slda = sda(processed_train[feature_names],processed_train$Best, stop = -sda_ft, K=min(2,A-1))
}

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
param.df = rbind(param.df, tibble(proj="SLDA",subclass=NA,
                                  nFeat=ifelse(cv,mod.slda$params$NumVars,sda_ft), eta=NA))

# ##### SMDA -----------
# if (cv) {
#   mod.smda = rcv.smda.par(formula_, data = metadata_train, K=5, reps = 2,
#                           msub=8,
#                           nvar.list = -1:-min(50, feat.rank, length(feature_names)),
#                           d=2, grid.size = 1000,
#                           scaler=scaler, n.cores=ncores)
#   best.smda = mod.smda$model
# } else {
#   mod.smda = NULL
#   best.smda = fit.smda(processed_train[feature_names],processed_train$Best, Rj=3, Q=2, nfeat=-sda_ft,
#            maxit=300, itmult=1,repInit=1, tol=1e-3, initMethod="kmeans")
# }
# 
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
# param.df = rbind(param.df, tibble(proj="SMDA",
#                                   subclass=ifelse(cv,list(best.smda$Rj),3),
#                                   nFeat=ifelse(cv,best.smda$nfeat,sda_ft), eta=NA))

###### write to csv -----
Z %>%
  mutate(across(starts_with("prob_"), ~ round(.x, 5))) %>%
  write_csv(paste0(outfol,"sda_proj.csv"))

Z.svm %>%
  mutate(across(starts_with("prob_"), ~ round(.x, 5))) %>%
  write_csv(paste0(outfol,"spls_proj.csv"))

param.df %>%
  mutate(subclass = sapply(subclass, toString)) %>%
  write_csv(paste0(outfol,"sparse_params.csv"))

### save objects
save(list = ls(pattern = "^(best\\.|mod\\.)"), file = paste0(outfol,"sparse_proj.Rdata"))

## update status if successful
file.create(paste0(outfol,"status.txt"))
fileConn<-file(paste0(outfol,"status.txt"))
writeLines("1", fileConn)
close(fileConn)
