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

######
#Grid
######

hidden_opt <- list(c(200,200,2000), c(500,500,2000), c(1000,2000), c(1000,1000))
act_opt <- c("TanhWithDropout","MaxoutWithDropout","RectifierWithDropout")
input_dropout_opt=c(0.1,0.2)
l1_opt <- c(1e-6, 1e-5)
stopping_rounds_opt <- c(0,3,4,5)
stopping_tol_opt <- c(1e-2, 1e-3, 1e-4)

hyper_params_1 <- list(hidden = hidden_opt, l1 = l1_opt)

model_grid_1 <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = 'grid_1',
  hyper_params = hyper_params_1,
  activation="RectifierWithDropout",
  input_dropout_ratio=0.1,
  x = x,
  y = y,
  distribution = "bernoulli", 
  training_frame = data.train,
  validation_frame = data.validate, 
  score_validation_samples = 0,
  epochs = 100,
  #stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 3 scoring events
  stopping_tolerance=1e-3,
  #stopping_tolerance=1e-4,
  stopping_rounds = 5,
  stopping_metric = 'AUC')

# print out all prediction errors and run times of the models
model_grid_1

# print out the Test AUC for all of the models
for (model_id in model_grid_1@model_ids) {
  auc <- h2o.auc(h2o.getModel(model_id), valid = TRUE)
  print(sprintf("Test set AUC: %f", auc))
}

##Up Epochs and lower stopping tol
model_grid_1a <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = 'grid_1a',
  hyper_params = hyper_params_1,
  activation="RectifierWithDropout",
  input_dropout_ratio=0.1,
  x = x,
  y = y,
  distribution = "bernoulli", 
  training_frame = data.train,
  validation_frame = data.validate, 
  score_validation_samples = 0,
  epochs = 200,
  #stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 3 scoring events
  #stopping_tolerance=1e-3,
  stopping_tolerance=1e-4,
  stopping_rounds = 5,
  stopping_metric = 'AUC')

# print out all prediction errors and run times of the models
model_grid_1a

# print out the Test AUC for all of the models
for (model_id in model_grid_1a@model_ids) {
  auc <- h2o.auc(h2o.getModel(model_id), valid = TRUE)
  print(sprintf("Test set AUC: %f", auc))
}

hyper_params_2 <- list(hidden = hidden_opt, l1 = l1_opt,
                       activation=act_opt,input_dropout_ratio=input_dropout_opt,
                       stopping_rounds=stopping_rounds_opt,
                       stopping_tolerance=stopping_tol_opt)

model_grid_2 <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = 'grid_2',
  hyper_params = hyper_params_2, 
  x = x,
  y = y,
  distribution = "bernoulli", 
  training_frame = data.train,
  validation_frame = data.validate, 
  score_validation_samples = 0,
  epochs = 100,
  stopping_metric = 'AUC')

# print out all prediction errors and run times of the models
model_grid_2

# print out the Test AUC for all of the models
for (model_id in model_grid_2@model_ids) {
  auc <- h2o.auc(h2o.getModel(model_id), valid = TRUE)
  print(sprintf("Test set AUC: %f", auc))
}


#####
#Save models
######


m1 <- h2o.getModel(model_grid_1@model_ids[[3]])
model_path_1 <- h2o.saveModel(object = m1, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
h2o.auc(m1, valid = TRUE)
print(model_path_1)

m2 <- h2o.getModel(model_grid_1@model_ids[[4]])
model_path_2 <- h2o.saveModel(object = m2, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
h2o.auc(m2, valid = TRUE)
print(model_path_2)
