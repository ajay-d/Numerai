rm(list=ls(all=TRUE))

library(h2o)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train <- read_csv("train_new.csv.gz")
test <- read_csv("test_new.csv.gz")

set.seed(2016)

######
#Score Data
######

load("deep.1.RData")

# print out the Test MSE for all of the models
for (model_id in model_grid_1@model_ids) {
  auc <- h2o.auc(h2o.getModel(model_id), valid = TRUE)
  print(sprintf("Test set AUC: %f", auc))
}

h2o.test <- test %>%
  select(-1) %>%
  as.h2o
deep.1 <- h2o.getModel(model_grid_1@model_ids[[3]])
pred <- h2o.predict(deep.1, newdata = h2o.test)

pred.deep <- as.data.frame(pred)
ggplot(pred.deep) + geom_histogram(aes(p1), binwidth = .001)

