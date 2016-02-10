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

deep_model_1.1 <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\grid_1_model_1")
deep_model_1.5 <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\grid_1_model_5")

test.h2o <- test %>%
  select(-ID) %>%
  as.h2o

pred_1 <- h2o.predict(deep_model_1.1, newdata = test.h2o)
pred_2 <- h2o.predict(deep_model_1.5, newdata = test.h2o)

ggplot(as.data.frame(pred_1)) + geom_histogram(aes(p1), binwidth = .001)
ggplot(as.data.frame(pred_1)) + geom_histogram(aes(p0), binwidth = .001)

ggplot(as.data.frame(pred_2)) + geom_histogram(aes(p1), binwidth = .001)

summary(as.data.frame(pred_1)$p1)
summary(as.data.frame(pred_2)$p1)
h2o.auc(deep_model_1.1, valid = TRUE)
h2o.auc(deep_model_1.5, valid = TRUE)

deep.1 <- as.data.frame(pred_1) %>%
  cbind(test %>% select(ID))  %>%
  select(t_id=ID, probability=p1)

deep.2.wrong <- as.data.frame(pred_2) %>%
  cbind(test %>% select(ID))  %>%
  select(t_id=ID, probability=p1)

deep.2 <- test %>%
  select(ID) %>%
  cbind(as.data.frame(pred_1) %>% select(p1=p1)) %>%
  cbind(as.data.frame(pred_2) %>% select(p2=p0)) %>%
  mutate(probability = (p1+p2)/2)

deep.blend.wrong <- test %>%
  select(ID) %>%
  cbind(as.data.frame(pred_1) %>% select(p1=p1)) %>%
  cbind(as.data.frame(pred_2) %>% select(p2=p1)) %>%
  mutate(probability = (p1+p2)/2)

ggplot(deep.blend.wrong) + geom_histogram(aes(probability), binwidth = .001)
ggplot(deep.1) + geom_histogram(aes(probability), binwidth = .001)

##AUC 0.5198
write.csv(deep.1,
          file='deep_1.csv', 
          row.names = FALSE)

##AUC 0.5166
write.csv(deep.2 %>%
            select(t_id=ID, probability),
          file='deep_2_blend.csv', 
          row.names = FALSE)

##AUC 0.5130
write.csv(deep.2.wrong,
          file='deep_2_wrong.csv', 
          row.names = FALSE)

##AUC 0.5190
write.csv(deep.blend.wrong %>%
            select(t_id=ID, probability),
          file='deep_blend_wrong.csv', 
          row.names = FALSE)
