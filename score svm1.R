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

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train <- read_csv("train_new.csv.gz")
test <- read_csv("test_new.csv.gz")

######
#GBM
######

gbm <- read_csv("gbm_2_3_newdata.csv")

######
#Score Data
######

load("svm.1.RData")

test.svm <- test %>%
  select(-ID) %>%
  as.matrix

svm.1 <- predict(csvm.1, test.svm)
svm.2 <- predict(csvm.2, test.svm)

svm.pred.1 <- test %>%
  select(ID) %>%
  bind_cols(as.data.frame(svm.1) %>% rename(svm.1=predictions)) %>%
  bind_cols(as.data.frame(svm.2) %>% rename(svm.2=predictions)) %>%
  rowwise() %>%
  mutate(svm.blend=sum(svm.1, svm.2)/2) %>%
  rename(t_id=ID) %>%
  inner_join(gbm) %>%
  mutate(probability2 = .98*probability + .02*svm.blend)

table(svm.pred.1$svm.1, svm.pred.1$svm.2)

svm.pred.2 <- test %>%
  select(ID) %>%
  bind_cols(as.data.frame(svm.1) %>% rename(svm.1=predictions)) %>%
  bind_cols(as.data.frame(svm.2) %>% rename(svm.2=predictions)) %>%
  rowwise() %>%
  mutate(svm.blend=sum(svm.1, svm.2)/2) %>%
  rename(t_id=ID) %>%
  inner_join(gbm) %>%
  mutate(probability2 = .95*probability + .05*svm.blend,
         probability3 = ifelse(svm.1==svm.2, .95*probability + .05*svm.blend, probability))

data <- svm.pred.1 %>%
  select(t_id, probability, probability2) %>%
  gather(variable, value, -t_id)

ggplot(data) + geom_histogram(aes(value, color=variable), binwidth = .001)

###AUC 0.5252
write.csv(svm.pred.1 %>%
            select(t_id, probability=probability2),
          file='gbm_svm_5pct.csv', 
          row.names = FALSE)
###AUC 0.5283
write.csv(svm.pred.1 %>%
            select(t_id, probability=probability2),
          file='gbm_svm_2pct.csv', 
          row.names = FALSE)

###AUC 0.5252
write.csv(svm.pred.2 %>%
            select(t_id, probability=probability3),
          file='gbm_svm_5pct_a.csv', 
          row.names = FALSE)


