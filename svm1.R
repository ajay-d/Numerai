rm(list=ls(all=TRUE))

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(R.utils)
library(SwarmSVM)

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

#train data only, no validation
train.1 <- train %>%
  select(ID, validation, target, one_of(col.names)) %>%
  inner_join(f.mn2)

#all training data
train.2 <- train %>%
  select(ID, validation, target, one_of(col.names)) %>%
  inner_join(f.mn1)

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


