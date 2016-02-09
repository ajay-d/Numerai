rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(xgboost)
library(R.utils)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

source("auc_functions.R")

train <- read_csv("train_new.csv.gz")
test <- read_csv("test_new.csv.gz")

set.seed(2016)

######
#Split Data
######

train.data.model.y <- train %>%
  filter(validation==0) %>%
  select(target) %>%
  as.matrix
train.data.watch.y <- train %>%
  filter(validation==1) %>%
  select(target) %>%
  as.matrix

train.data.model <- train %>%
  filter(validation==0) %>%
  select(-ID, -validation, -target) %>%
  as.matrix
train.data.watch <- train %>%
  filter(validation==1) %>%
  select(-ID, -validation, -target) %>%
  as.matrix

dtrain <- xgb.DMatrix(data = train.data.model, label = train.data.model.y)
dtest <- xgb.DMatrix(data = train.data.watch, label = train.data.watch.y)

watchlist <- list(train=dtrain, test=dtest)

bst.1 <- xgb.train(data = dtrain, 
                   watchlist = watchlist,
                   max.depth = 10, eta = .04, nround = 50,
                   verbose = 1,
                   nthread = 8,
                   feval = auc.gbm, 
                   maximize = TRUE,
                   objective = "binary:logistic")
bst.2 <- xgb.train(data = dtrain, 
                   watchlist = watchlist,
                   max.depth = 9, eta = .03, nround = 50,
                   verbose = 1,
                   nthread = 8,
                   eval_metric = "auc",
                   objective = "binary:logistic")

xgb.1 <- predict(bst.1, train.data.watch)
xgb.2 <- predict(bst.2, train.data.watch)

ggplot(as.data.frame(xgb.1)) + geom_histogram(aes(xgb.1), binwidth = .001)
ggplot(as.data.frame(xgb.2)) + geom_histogram(aes(xgb.2), binwidth = .001)

quantile(xgb.1, c(0,.01, .05, .25, .5, .75, .95, .99, 1))
quantile(xgb.2, c(0,.01, .05, .25, .5, .75, .95, .99, 1))

auc(train.data.watch, xgb.1)
auc(train.data.watch, xgb.2)

###Full Data

train.data.model.y <- train %>%
  select(target) %>%
  as.matrix

train.data.model <- train %>%
  select(-ID, -validation, -target) %>%
  as.matrix

dtrain <- xgb.DMatrix(data = train.data.model, label = train.data.model.y)

bst.3 <- xgb.train(data = dtrain, 
                   watchlist = watchlist,
                   max.depth = 10, eta = .04, nround = 50,
                   verbose = 1,
                   nthread = 8,
                   feval = auc.gbm, 
                   maximize = TRUE,
                   objective = "binary:logistic")
bst.4 <- xgb.train(data = dtrain, 
                   max.depth = 9, eta = .03, nround = 50,
                   verbose = 1,
                   nthread = 8,
                   feval = auc.gbm, 
                   maximize = TRUE,
                   objective = "binary:logistic")

xgb.3 <- predict(bst.3, train.data.watch)
xgb.4 <- predict(bst.4, train.data.watch)

ggplot(as.data.frame(xgb.3)) + geom_histogram(aes(xgb.3), binwidth = .001)
ggplot(as.data.frame(xgb.4)) + geom_histogram(aes(xgb.4), binwidth = .001)

quantile(xgb.3, c(0,.01, .05, .25, .5, .75, .95, .99, 1))
quantile(xgb.4, c(0,.01, .05, .25, .5, .75, .95, .99, 1))

auc(train.data.watch, xgb.3)
auc(train.data.watch, xgb.4)

######
#Score Data
######

test.gbm <- test %>%
  select(-ID) %>%
  as.matrix
