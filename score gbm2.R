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

train.full <- read_csv("numerai_datasets/numerai_training_data.csv")
test.full <- read_csv("numerai_datasets/numerai_tournament_data.csv")
sample.submission <- read_csv("numerai_datasets/numerai_example_predictions.csv")

train.full %>% count(c1)
test.full %>% count(c1)

train.full %>% count(target)
train.full %>% count(validation, target)

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
#Test Data
######

f.mn <- test.full %>%
  select(starts_with("f"), t_id) %>%
  gather(variable, value, -t_id) %>% 
  left_join(mn.variables) %>%
  mutate(f.m1 = (value-m.1)/sd.1,
         f.m2 = (value-m.2)/sd.2,
         f.m3 = (value-m.3)/sd.3,
         f.m4 = (value-m.4)/sd.4)

f.names <- test.full %>% select(starts_with("f")) %>% names

f.mn1 <- f.mn %>%
  select(t_id, variable, f.m1) %>%
  spread(variable, f.m1) %>%
  #reorder column names
  select(t_id, one_of(f.names)) %>%
  setNames(c('t_id', paste0(f.names, '.m1')))
f.mn2 <- f.mn %>%
  select(t_id, variable, f.m2) %>%
  spread(variable, f.m2) %>%
  #reorder column names
  select(t_id, one_of(f.names)) %>%
  setNames(c('t_id', paste0(f.names, '.m2')))
f.mn3 <- f.mn %>%
  select(t_id, variable, f.m3) %>%
  spread(variable, f.m3) %>%
  #reorder column names
  select(t_id, one_of(f.names)) %>%
  setNames(c('t_id', paste0(f.names, '.m3')))
f.mn4 <- f.mn %>%
  select(t_id, variable, f.m4) %>%
  spread(variable, f.m4) %>%
  #reorder column names
  select(t_id, one_of(f.names)) %>%
  setNames(c('t_id', paste0(f.names, '.m4')))

f.all <- f.mn1 %>%
  inner_join(f.mn2) %>%
  inner_join(f.mn3) %>%
  inner_join(f.mn4)

col.names <- paste("c_", 1:5, sep='')

test <- test.full %>%
  separate(c1, into = c('c2', 'c.num'), remove=FALSE, convert=TRUE) %>%
  mutate(bin = intToBin(c.num)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)

test <- test %>%
  select(t_id, one_of(col.names)) %>%
  inner_join(f.all)

######
#Score Data
######

load("gbm.2.RData")

test.gbm <- test %>%
  select(-t_id) %>%
  as.matrix

xgb.2 <- predict(bst.2, test.gbm)
xgb.3 <- predict(bst.3, test.gbm)

gbm.pred.1 <- test %>%
  select(t_id) %>%
  bind_cols(as.data.frame(xgb.2)) %>%
  bind_cols(as.data.frame(xgb.3)) %>%
  rowwise() %>%
  mutate(probability=sum(xgb.2, xgb.3)/2)

data <- gbm.pred.1 %>%
  gather(variable, value, -t_id)
ggplot(data) + geom_histogram(aes(value, color=variable), binwidth = .001)
ggplot(gbm.pred.1) + geom_histogram(aes(xgb.2), binwidth = .001)
ggplot(gbm.pred.1) + geom_histogram(aes(xgb.3), binwidth = .001)

###AUC 0.5242
write.csv(gbm.pred.1 %>%
            select(t_id, probability),
          file='gbm_2_3.csv', 
          row.names = FALSE)




