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
#GBM
######

gbm.1 <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_1")
gbm.1f <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_1_full")

gbm.2 <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_2")
gbm.2f <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_2_full")

gbm.3 <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_3")
gbm.3f <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_3_full")

gbm.4 <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_4")
gbm.4f <- h2o.loadModel("C:\\Users\\adeonari\\Downloads\\Numerai\\gbm_4_full")

test.h2o <- test %>%
  select(-ID) %>%
  as.h2o

pred_1 <- h2o.predict(gbm.1, newdata = test.h2o)
pred_1f <- h2o.predict(gbm.1f, newdata = test.h2o)

pred_2 <- h2o.predict(gbm.2, newdata = test.h2o)
pred_2f <- h2o.predict(gbm.2f, newdata = test.h2o)

pred_3 <- h2o.predict(gbm.3, newdata = test.h2o)
pred_3f <- h2o.predict(gbm.3f, newdata = test.h2o)

pred_4 <- h2o.predict(gbm.4, newdata = test.h2o)
pred_4f <- h2o.predict(gbm.4f, newdata = test.h2o)

ggplot(as.data.frame(pred_1)) + geom_histogram(aes(p1), binwidth = .001)
ggplot(as.data.frame(pred_2)) + geom_histogram(aes(p1), binwidth = .001)

gbm_all <- cbind(test %>% select(ID)) %>%
  cbind(as.data.frame(pred_1) %>% select(p1=p1)) %>%
  cbind(as.data.frame(pred_1f) %>% select(p1f=p1)) %>%
  cbind(as.data.frame(pred_2) %>% select(p2=p1)) %>%
  cbind(as.data.frame(pred_2f) %>% select(p2f=p1)) %>%
  cbind(as.data.frame(pred_3) %>% select(p3=p1)) %>%
  cbind(as.data.frame(pred_3f) %>% select(p3f=p1)) %>%
  cbind(as.data.frame(pred_4) %>% select(p4=p1)) %>%
  cbind(as.data.frame(pred_4f) %>% select(p4f=p1))

h2o.blend.4 <- gbm_all %>%
  mutate(probability = (p1+p1f+p2+p2f) / 4) %>%
  select(t_id=ID, probability)

h2o.blend.8 <- gbm_all %>%
  mutate(probability = (p1+p1f+p2+p2f+p3+p3f+p4+p4f) / 8) %>%
  select(t_id=ID, probability)

ggplot(h2o.blend.4) + geom_histogram(aes(probability), binwidth = .001)
ggplot(h2o.blend.8) + geom_histogram(aes(probability), binwidth = .001)

summary(h2o.blend.4$probability)
summary(h2o.blend.8$probability)

##AUC 0.5291
write.csv(h2o.blend.4,
          file='h2o_gbm_blend4.csv', 
          row.names = FALSE)

##AUC 0.5296
write.csv(h2o.blend.8,
          file='h2o_gbm_blend8.csv', 
          row.names = FALSE)

####################################################################

gbm.blend.3 <- read.csv("gbm_blend_3.csv")
h2o.blend.8 <- read.csv("h2o_gbm_blend8.csv")

gbm.blendw.h2o <- gbm.blend.3 %>%
  select(t_id, prob1=probability) %>%
  inner_join(h2o.blend.8) %>%
  rename(prob2=probability) %>%
  mutate(probability = (prob1+prob2)/2) %>%
  select(t_id, probability)

##AUC 0.5329
write.csv(gbm.blendw.h2o,
          file='h2o_gbm_blend8_and_gbm_blend_3.csv', 
          row.names = FALSE)


