rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(R.utils)
library(SwarmSVM)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train.full <- read_csv("train_new.csv.gz")
test.full <- read_csv("test_new.csv.gz")

set.seed(2016)

######
#SVM
######

train.data.svm <- train %>%
  filter(validation==0) %>%
  select(-ID, -validation) %>%
  as.matrix

test.data.svm <- train %>%
  filter(validation==1) %>%
  select(-ID, -validation) %>%
  as.matrix

csvm.1 = clusterSVM(x = train.data.svm[,-1], y = train.data.svm[,1], type = 1,
                    valid.x = test.data.svm[,-1],valid.y = test.data.svm[,1], 
                    seed = 2016, verbose = 1, centers = 6)

csvm.2 = clusterSVM(x = train.data.svm[,-1], y = train.data.svm[,1], type = 1,
                    valid.x = test.data.svm[,-1],valid.y = test.data.svm[,1],
                    cluster.method = "mlKmeans",
                    seed = 2016, verbose = 1, centers = 4)

#Early Stop
dcsvm.1 = dcSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                k = 4, max.levels = 4, seed = 2016, 
                kernel = 3, early = 1, m = 800,
                #proba = TRUE,
                cluster.method = "mlKmeans",
                valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1])

dcsvm.2 = dcSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                k = 4, max.levels = 5, seed = 2016,
                kernel = 3, early = 1, m = 800,
                #proba = TRUE,
                cluster.method = "mlKmeans",
                valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1])

csvm.1$valid.score
csvm.2$valid.score
dcsvm.1$valid.score
dcsvm.2$valid.score

preds = dcsvm.3$valid.pred
table(preds, test.data.svm[,1])

preds = csvm.1$valid.pred
table(preds, test.data.svm[,1])

gaterSVM.1 = gaterSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                      hidden = 10, seed = 2016,
                      m = 10, max.iter = 3, learningrate = 0.01, threshold = 1, stepmax = 1000,
                      valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1], verbose = TRUE)

gaterSVM.1$valid.score
