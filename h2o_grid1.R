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
#GBM
######

h2o.gbm.1 <- h2o.gbm(y = y, 
                     x = x, 
                     model_id='gbm_1',
                     distribution="bernoulli",
                     training_frame = data.train,
                     validation_frame = data.validate,
                     ntrees=100, 
                     max_depth=4, 
                     learn_rate=0.1)

h2o.gbm.1
h2o.auc(h2o.gbm.1, valid = TRUE)
h2o.auc(h2o.gbm.1, train = TRUE)

######
#Grid
######

ntrees_opt <- c(50,75,100)
maxdepth_opt <- c(4,5,6,7,8,9,10)
learnrate_opt <- c(.01,.05,0.1,0.2)
hyper_parameters <- list(ntrees=ntrees_opt, 
                         max_depth=maxdepth_opt, 
                         learn_rate=learnrate_opt)

gbm.grid.1 <- h2o.grid("gbm", 
                       grid_id = 'gbm_grid_1',
                       hyper_params = hyper_parameters, 
                       y = y, 
                       x = x, 
                       distribution="bernoulli",
                       training_frame = data.train,
                       validation_frame = data.validate)

grid <- NULL
for (model_id in gbm.grid.1@model_ids) {
  auc <- h2o.auc(h2o.getModel(model_id), valid = TRUE)
  
  ntrees <- h2o.getModel(model_id)@allparameters$ntrees
  max_depth <- h2o.getModel(model_id)@allparameters$max_depth
  learn_rate <- h2o.getModel(model_id)@allparameters$learn_rate
  
  df <- data_frame(auc=auc,
                   model_id=model_id,
                   ntrees=ntrees,           
                   max_depth=max_depth,
                   learn_rate=learn_rate)
  grid <- bind_rows(grid, df)
}

grid <- grid %>%
  arrange(desc(auc))

best.1 <- h2o.getModel(grid[[1, 'model_id']])
best.2 <- h2o.getModel(grid[[2, 'model_id']])
best.3 <- h2o.getModel(grid[[3, 'model_id']])
best.4 <- h2o.getModel(grid[[4, 'model_id']])


######
#Run best models w/full and train data
######

h2o.gbm.1 <- h2o.gbm(y = y, 
                     x = x, 
                     model_id='gbm_1',
                     distribution="bernoulli",
                     training_frame = data.train,
                     validation_frame = data.validate,
                     ntrees=100, 
                     max_depth=4, 
                     learn_rate=0.05)

h2o.auc(h2o.gbm.1, valid = TRUE)

h2o.gbm.1.full <- h2o.gbm(y = y, 
                          x = x, 
                          model_id='gbm_1_full',
                          distribution="bernoulli",
                          training_frame = data.train.all,
                          ntrees=100, 
                          max_depth=4, 
                          learn_rate=0.05)

h2o.auc(h2o.gbm.1.full, train = TRUE)

h2o.gbm.2 <- h2o.gbm(y = y, 
                     x = x, 
                     model_id='gbm_2',
                     distribution="bernoulli",
                     training_frame = data.train,
                     validation_frame = data.validate,
                     ntrees=75, 
                     max_depth=4, 
                     learn_rate=0.05)

h2o.auc(h2o.gbm.2, valid = TRUE)

h2o.gbm.2.full <- h2o.gbm(y = y, 
                          x = x, 
                          model_id='gbm_2_full',
                          distribution="bernoulli",
                          training_frame = data.train.all,
                          ntrees=75, 
                          max_depth=4, 
                          learn_rate=0.05)

h2o.auc(h2o.gbm.2.full, train = TRUE)

h2o.gbm.3 <- h2o.gbm(y = y, 
                     x = x, 
                     model_id='gbm_3',
                     distribution="bernoulli",
                     training_frame = data.train,
                     validation_frame = data.validate,
                     ntrees=100, 
                     max_depth=5, 
                     learn_rate=0.05)

h2o.auc(h2o.gbm.3, valid = TRUE)

h2o.gbm.3.full <- h2o.gbm(y = y, 
                          x = x, 
                          model_id='gbm_3_full',
                          distribution="bernoulli",
                          training_frame = data.train.all,
                          ntrees=100, 
                          max_depth=5, 
                          learn_rate=0.05)

h2o.auc(h2o.gbm.3.full, train = TRUE)

h2o.gbm.4 <- h2o.gbm(y = y, 
                     x = x, 
                     model_id='gbm_4',
                     distribution="bernoulli",
                     training_frame = data.train,
                     validation_frame = data.validate,
                     ntrees=50, 
                     max_depth=4, 
                     learn_rate=0.05)

h2o.auc(h2o.gbm.4, valid = TRUE)

h2o.gbm.4.full <- h2o.gbm(y = y, 
                          x = x, 
                          model_id='gbm_4_full',
                          distribution="bernoulli",
                          training_frame = data.train.all,
                          ntrees=50, 
                          max_depth=4, 
                          learn_rate=0.05)

h2o.auc(h2o.gbm.4.full, train = TRUE)

#####
#Save models
######

model_gbm_1 <- h2o.saveModel(object = h2o.gbm.1, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_1_full <- h2o.saveModel(object = h2o.gbm.1.full, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_2 <- h2o.saveModel(object = h2o.gbm.2, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_2_full <- h2o.saveModel(object = h2o.gbm.2.full, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_3 <- h2o.saveModel(object = h2o.gbm.3, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_3_full <- h2o.saveModel(object = h2o.gbm.3.full, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_4 <- h2o.saveModel(object = h2o.gbm.4, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
model_gbm_4_full <- h2o.saveModel(object = h2o.gbm.4.full, path="C:\\Users\\adeonari\\Downloads\\Numerai", force = TRUE)
