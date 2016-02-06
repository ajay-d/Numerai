rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(xgboost)
library(R.utils)
library(SwarmSVM)
library(rotationForest)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train.full <- read_csv("numerai_datasets/numerai_training_data.csv")
test.full <- read_csv("numerai_datasets/numerai_tournament_data.csv")
sample.submission <- read_csv("numerai_datasets/numerai_example_predictions.csv")

train.full %>% count(c1)
test.full %>% count(c1)

train.full %>% count(target)
train.full %>% count(validation, target)

features <- train.full %>%
  select(starts_with("f"))

f.cor <- cor(features, use="pairwise.complete.obs")
f.cor <- apply(f.cor, 2, function (x) ifelse(x==1,0,x))
f.cor.max <- apply(f.cor, 1, max)
f.cor.max

ggplot(sample.submission) + geom_histogram(aes(probability), binwidth = .001)

ggplot(train.full) + geom_histogram(aes(f1))
ggplot(train.full) + geom_histogram(aes(f2))

set.seed(2016)

train <- train.full %>%
  mutate(ID = row_number()) %>%
  separate(c1, into = c('c2', 'c.num'), remove=FALSE, convert=TRUE)

train %>% summarise(max(c.num))

######
#How to scale data
######

mn.variables <- train %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  summarise(m.1 = mean(value),
            sd.1 = sd(value)) %>%
  inner_join(train %>%
              filter(validation==0) %>%
              select(starts_with("f"), ID) %>%
              gather(variable, value, -ID) %>% 
              group_by(variable) %>%
              summarise(m.2 = mean(value),
                        sd.2 = sd(value))) %>%
  inner_join(train %>%
              filter(validation==1) %>%
              select(starts_with("f"), ID) %>%
              gather(variable, value, -ID) %>% 
              group_by(variable) %>%
              summarise(m.3 = mean(value),
                        sd.3 = sd(value))) %>%
  inner_join(train.full %>%
              bind_rows(test.full) %>%
              select(starts_with("f")) %>%
              gather(variable, value) %>% 
              group_by(variable) %>%
              summarise(m.4 = mean(value),
                        sd.4 = sd(value)))
  
f.mn <- train %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  left_join(mn.variables) %>%
  mutate(f.m1 = (value-m.1)/sd.1,
         f.m2 = (value-m.2)/sd.2,
         f.m3 = (value-m.3)/sd.3,
         f.m4 = (value-m.4)/sd.4)

f.names <- train.full %>% select(starts_with("f")) %>% names

f.mn1 <- f.mn %>%
  select(ID, variable, f.m1) %>%
  spread(variable, f.m1) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.m1')))
f.mn2 <- f.mn %>%
  select(ID, variable, f.m2) %>%
  spread(variable, f.m2) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.m2')))
f.mn3 <- f.mn %>%
  select(ID, variable, f.m3) %>%
  spread(variable, f.m3) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.m3')))
f.mn4 <- f.mn %>%
  select(ID, variable, f.m4) %>%
  spread(variable, f.m4) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.m4')))

f.all <- f.mn1 %>%
  inner_join(f.mn2) %>%
  inner_join(f.mn3) %>%
  inner_join(f.mn4)

intToBin(24)
col.names <- paste("c_", 1:5, sep='')
train <- train %>%
  mutate(bin = intToBin(c.num)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)

train <- train %>%
  select(ID, validation, target, one_of(col.names)) %>%
  inner_join(f.all)


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
                   max.depth = 25, eta = .05, nround = 50,
                   verbose = 1,
                   nthread = 8,
                   objective = "binary:logistic")
bst.2 <- xgb.train(data = dtrain, 
                   watchlist = watchlist,
                   max.depth = 10, eta = .04, nround = 50,
                   verbose = 1,
                   nthread = 8,
                   objective = "binary:logistic")

xgb.1 <- predict(bst.1, train.data.watch)
xgb.2 <- predict(bst.2, train.data.watch)

ggplot(as.data.frame(xgb.1)) + geom_histogram(aes(xgb.1), binwidth = .001)
ggplot(as.data.frame(xgb.2)) + geom_histogram(aes(xgb.2), binwidth = .001)

######
#Rotation Forest
######

train.data.rf <- train %>%
  filter(validation==0) %>%
  select(-ID, -validation) %>%
  mutate(target=as.factor(target)) %>%
  select(target, starts_with("c_"), contains("m1"))

test.data.rf <- train %>%
  filter(validation==1) %>%
  select(-ID, -validation) %>%
  select(starts_with("c_"), contains("m1"))

rf.1 <- rotationForest(x=train.data.rf %>% select(-target), 
                       y=train.data.rf$target, 
                       L = 10)
rf.2 <- rotationForest(x=train.data.rf %>% select(-target), 
                       y=train.data.rf$target, 
                       L = 50)

rf.1 <- predict(object=rf.1, newdata=test.data.rf)
rf.2 <- predict(object=rf.2, newdata=test.data.rf)

ggplot(as.data.frame(rf.1)) + geom_histogram(aes(rf.1), binwidth = .001)
ggplot(as.data.frame(rf.2)) + geom_histogram(aes(rf.2), binwidth = .001)

summary(rf.1)
summary(rf.2)

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

csvm.obj = clusterSVM(x = train.data.svm[,-1], y = train.data.svm[,1], type = 1,
                      valid.x = test.data.svm[,-1],valid.y = test.data.svm[,1], 
                      seed = 2016, verbose = 1, centers = 8)

csvm.obj$valid.score
summary(csvm.obj)

#Don't Stop Early
##the kernel type: 1 for linear, 2 for polynomial, 3 for gaussian
dcsvm.0 = dcSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                k = 4, max.levels = 4, seed = 2016,
                ##Try defaults
                cost = 32, gamma = 2,
                kernel = 3, early = 0, m = 800,
                valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1])
#Early Stop
dcsvm.1 = dcSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                k = 4, max.levels = 4, seed = 2016, 
                cost = 32, gamma = 2,
                kernel = 3, early = 1, m = 800,
                #proba = TRUE,
                cluster.method = "mlKmeans",
                valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1])
#Exact
dcsvm.2 = dcSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                k = 10, max.levels = 1, 
                cost = 32, gamma = 2,
                kernel = 2, early = 1, tolerance = 1e-2, m = 800, 
                valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1])

dcsvm.0$valid.score
dcsvm.1$valid.score
dcsvm.2$valid.score

preds = dcsvm.0$valid.pred
table(preds, test.data.svm[,1])

preds = dcsvm.1$valid.pred
table(preds, test.data.svm[,1])

gaterSVM.1 = gaterSVM(x = train.data.svm[,-1], y = train.data.svm[,1],
                      hidden = 10, seed = 2016,
                      m = 10, max.iter = 3, learningrate = 0.01, threshold = 1, stepmax = 1000,
                      valid.x = test.data.svm[,-1], valid.y = test.data.svm[,1], verbose = TRUE)

gaterSVM.1$valid.score

######
#Score
######

pred.1 <- train %>%
  filter(validation==1) %>%
  select(ID, target) %>%
  bind_cols(as.data.frame(gbm.1)) %>%
  bind_cols(as.data.frame(xgb.1)) %>%
  bind_cols(as.data.frame(xgb.2))

save(bst.1, bst.2, file='gbm.1.RData')
