rm(list=ls(all=TRUE))

library(h2o)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

localH2O <- h2o.init(ip = 'localhost', nthreads=10, max_mem_size = '64g')
#h2o.clusterInfo()

#train <- h2o.importFile("train_new.csv.gz")
#test <- h2o.importFile("test_new.csv.gz")
#summary(train)

train <- read_csv("train_new.csv.gz")
test <- read_csv("test_new.csv.gz")

set.seed(2016)

######
#Deep
######

data.train <- train %>%
  filter(validation==0) %>%
  select(-ID, -validation) %>%
  #select(-ID, -validation, target, contains("C_"), -contains("prod"), -contains("sq")) %>%
  mutate(target=as.factor(target)) %>%
  as.h2o
data.validate <- train %>%
  filter(validation==1) %>%
  select(-ID, -validation) %>%
  #select(-ID, -validation, target, contains("C_"), -contains("prod"), -contains("sq")) %>%
  mutate(target=as.factor(target)) %>%
  as.h2o

x <- setdiff(names(data.train), 'target')
y <- 'target'

data.train.all <- train %>%
  select(-ID, -validation) %>%
  mutate(target=as.factor(target)) %>%
  as.h2o

x.all <- setdiff(names(data.train.all), 'target')
y.all <- 'target'

######
#Single Fulls and Partials
######

data.train <- train %>%
  filter(validation==0) %>%
  select(-ID, -validation) %>%
  #select(-ID, -validation, target, contains("C_"), -contains("prod"), -contains("sq")) %>%
  mutate(target=as.factor(target)) %>%
  as.h2o
data.validate <- train %>%
  filter(validation==1) %>%
  select(-ID, -validation) %>%
  #select(-ID, -validation, target, contains("C_"), -contains("prod"), -contains("sq")) %>%
  mutate(target=as.factor(target)) %>%
  as.h2o

x <- setdiff(names(data.train), 'target')
y <- 'target'

data.train.all <- train %>%
  select(-ID, -validation) %>%
  mutate(target=as.factor(target)) %>%
  as.h2o

model.1 <- h2o.deeplearning(
  x = x, 
  y = y, 
  model_id = "model_1_train",
  training_frame = data.train,
  validation_frame = data.validate,
 	activation="RectifierWithDropout", 
 	hidden=c(200,200,2000), 
  epochs=100, 
  
 	input_dropout_ratio=0.1,     ##Specifies the fraction of the features for each training row to omit from training to improve generalization. 
                               ##The defaultis 0, which always uses all features
  l1=1e-5,                     ##The default is 0, for no L1 regularization
 	score_validation_samples=0,
  stopping_rounds=5, 
  stopping_metric="AUC",
  stopping_tolerance=0.001
  )

##AUC 
h2o.auc(model.1, valid = TRUE)
h2o.auc(model.1, train = TRUE)

model.1.full <- h2o.deeplearning(
  x = x, 
  y = y, 
  model_id = "model_1_full",
  training_frame = data.train.all,
 	activation="RectifierWithDropout", 
 	hidden=c(200,200,2000), 
  epochs=100, 
 	
  input_dropout_ratio=0.1,     ##Specifies the fraction of the features for each training row to omit from training to improve generalization. 
                               ##The defaultis 0, which always uses all features
  l1=1e-5,                     ##The default is 0, for no L1 regularization

  stopping_rounds=5, 
  stopping_metric="AUC",
  stopping_tolerance=0.001
  )

##AUC 
h2o.auc(model.1.full, train = TRUE)