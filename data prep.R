rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
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
train.full %>% count(target)
train.full %>% count(validation, target)

features <- train.full %>%
  select(starts_with("f"))

f.cor <- cor(features, use="pairwise.complete.obs")
f.cor <- apply(f.cor, 2, function (x) ifelse(x==1,0,x))
f.cor.max <- apply(f.cor, 1, max)
f.cor.max

ggplot(train.full) + geom_histogram(aes(f1))
ggplot(train.full) + geom_histogram(aes(f2))

set.seed(2016)

train <- train.full %>%
  mutate(ID = row_number()) %>%
  separate(c1, into = c('c2', 'c.num'), remove=FALSE, convert=TRUE)

train %>% summarise(max(c.num))

mn.1 <- train %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  mutate(m = mean(value),
         sd = sd(value),
         f.m1 = (value-m)/sd) %>%
  select(ID, f.m1) %>%
  spread(variable, f.m1)

mn.2 <- train %>%
  filter(validation==1) %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  mutate(m = mean(value),
         sd = sd(value),
         f.m2 = (value-m)/sd) %>%
  select(ID, f.m2) %>%
  spread(variable, f.m2)
  
mn.3 <- train %>%
  filter(validation==0) %>%
  select(starts_with("f"), ID) %>%
  gather(variable, value, -ID) %>% 
  group_by(variable) %>%
  mutate(m = mean(value),
         sd = sd(value),
         f.m3 = (value-m)/sd) %>%
  select(ID, f.m3) %>%
  spread(variable, f.m3)

intToBin(24)
col.names <- paste("c_", 1:5, sep='')
train <- train %>%
  mutate(bin = intToBin(c.num)) %>%
  rowwise() %>%
  mutate(bin.split = paste0(strsplit(bin, split='')[[1]], collapse = ".")) %>%
  separate(bin.split, col.names, convert=TRUE)

train <- train %>%
  select(ID, validation, target, one_of(col.names)) %>%
  inner_join(mn.1)




