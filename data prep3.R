rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
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

features <- train.full %>%
  select(starts_with("f"))

f.cor <- cor(features, use="pairwise.complete.obs")
f.cor <- apply(f.cor, 2, function (x) ifelse(x==1,0,x))
f.cor.max <- apply(f.cor, 1, max)
f.cor.max

apply(f.cor, 2, mean)

set.seed(2016)

train <- train.full %>%
  mutate(ID = row_number()) %>%
  separate(c1, into = c('c2', 'c.num'), remove=FALSE, convert=TRUE)

train %>% summarise(max(c.num))

######
#Scale Training
######

f.names <- train.full %>% select(starts_with("f")) %>% names

mn.train.0 <- train %>%
  filter(validation==0) %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  mutate(value2 = as.double(value^2),
         m.1 = mean(value),
         sd.1 = sd(value),
         m.2 = mean(value2),
         sd.2 = sd(value2),
         f.mn = (value-m.1)/sd.1,
         f2.mn = (value2-m.2)/sd.2)

mn.train.0.sq <- mn.train.0 %>%
  select(ID, variable, f2.mn) %>%
  spread(variable, f2.mn) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.mn.sq')))

mn.train.0 <- mn.train.0 %>%
  select(ID, variable, f.mn) %>%
  spread(variable, f.mn) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.mn')))

mn.train.0.f5 <- train %>%
  filter(validation==0) %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>%
  filter(variable != 'f5') %>%
  inner_join(train %>%
               filter(validation==0) %>%
               select(f5, ID) %>%
               gather(f5.var, f5.value, -ID)) %>%
  mutate(f5.prod = as.double(value)*as.double(f5.value)) %>%
  group_by(variable) %>%
  mutate(m.1 = mean(f5.prod),
         sd.1 = sd(f5.prod),
         fprod.mn = (f5.prod-m.1)/sd.1)

mn.train.0.f5 <- mn.train.0.f5 %>%
  select(ID, variable, fprod.mn) %>%
  spread(variable, fprod.mn) %>%
  #reorder column names
  select(ID, one_of(setdiff(f.names, 'f5'))) %>%
  setNames(c('ID', paste0(setdiff(f.names, 'f5'), '.prod')))

mn.train.0 <- mn.train.0 %>%
  inner_join(mn.train.0.sq) %>%
  inner_join(mn.train.0.f5)

mn.train.1 <- train %>%
  filter(validation==1) %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  mutate(value2 = as.double(value^2),
         m.1 = mean(value),
         sd.1 = sd(value),
         m.2 = mean(value2),
         sd.2 = sd(value2),
         f.mn = (value-m.1)/sd.1,
         f2.mn = (value2-m.2)/sd.2)

mn.train.1.sq <- mn.train.1 %>%
  select(ID, variable, f2.mn) %>%
  spread(variable, f2.mn) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.mn.sq')))

mn.train.1 <- mn.train.1 %>%
  select(ID, variable, f.mn) %>%
  spread(variable, f.mn) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.mn')))

mn.train.1.f5 <- train %>%
  filter(validation==1) %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>%
  filter(variable != 'f5') %>%
  inner_join(train %>%
               filter(validation==1) %>%
               select(f5, ID) %>%
               gather(f5.var, f5.value, -ID)) %>%
  mutate(f5.prod = as.double(value)*as.double(f5.value)) %>%
  group_by(variable) %>%
  mutate(m.1 = mean(f5.prod),
         sd.1 = sd(f5.prod),
         fprod.mn = (f5.prod-m.1)/sd.1)

mn.train.1.f5 <- mn.train.1.f5 %>%
  select(ID, variable, fprod.mn) %>%
  spread(variable, fprod.mn) %>%
  #reorder column names
  select(ID, one_of(setdiff(f.names, 'f5'))) %>%
  setNames(c('ID', paste0(setdiff(f.names, 'f5'), '.prod')))

mn.train.1 <- mn.train.1 %>%
  inner_join(mn.train.1.sq) %>%
  inner_join(mn.train.1.f5)

train.mn <- bind_rows(mn.train.0, mn.train.1) %>%
  arrange(ID)

intToBin(24)
col.names <- paste("c_", 1:5, sep='')
train <- train %>%
  mutate(bin = intToBin(c.num)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)

train <- train %>%
  select(ID, validation, target, one_of(col.names)) %>%
  inner_join(train.mn)

######
#Scale Testing
######

test <- test.full %>%
  rename(ID = t_id) %>%
  separate(c1, into = c('c2', 'c.num'), remove=FALSE, convert=TRUE)

mn.test <- test %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  mutate(value2 = as.double(value^2),
         m.1 = mean(value),
         sd.1 = sd(value),
         m.2 = mean(value2),
         sd.2 = sd(value2),
         f.mn = (value-m.1)/sd.1,
         f2.mn = (value2-m.2)/sd.2)

mn.test.sq <- mn.test %>%
  select(ID, variable, f2.mn) %>%
  spread(variable, f2.mn) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.mn.sq')))

mn.test <- mn.test %>%
  select(ID, variable, f.mn) %>%
  spread(variable, f.mn) %>%
  #reorder column names
  select(ID, one_of(f.names)) %>%
  setNames(c('ID', paste0(f.names, '.mn')))

mn.test.f5 <- test %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>%
  filter(variable != 'f5') %>%
  inner_join(test %>%
               select(f5, ID) %>%
               gather(f5.var, f5.value, -ID)) %>%
  mutate(f5.prod = as.double(value)*as.double(f5.value)) %>%
  group_by(variable) %>%
  mutate(m.1 = mean(f5.prod),
         sd.1 = sd(f5.prod),
         fprod.mn = (f5.prod-m.1)/sd.1)

mn.test.f5 <- mn.test.f5 %>%
  select(ID, variable, fprod.mn) %>%
  spread(variable, fprod.mn) %>%
  #reorder column names
  select(ID, one_of(setdiff(f.names, 'f5'))) %>%
  setNames(c('ID', paste0(setdiff(f.names, 'f5'), '.prod')))

test.mn <- mn.test %>%
  inner_join(mn.test.sq) %>%
  inner_join(mn.test.f5)

col.names <- paste("c_", 1:5, sep='')
test <- test %>%
  mutate(bin = intToBin(c.num)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)

test <- test %>%
  select(ID, one_of(col.names)) %>%
  inner_join(test.mn)

write.csv(test, gzfile('test_new.csv.gz'), row.names=FALSE)
write.csv(train, gzfile('train_new.csv.gz'), row.names=FALSE)
