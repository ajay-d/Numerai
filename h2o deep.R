rm(list=ls(all=TRUE))

library(h2o)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

options(mc.cores = parallel::detectCores(),
        stringsAsFactors = FALSE,
        scipen = 10) 

#' coalesce function
`%||%` <- function(a, b) ifelse(!is.na(a), a, b)

train <- read_csv("train_new.csv.gz")
test <- read_csv("test_new.csv.gz")

######
#Deep
######

localH2O <- h2o.init(ip = 'localhost', nthreads=6, max_mem_size = '64g')
h2o.clusterInfo()

h2o.train <- train %>%
  select(-ID, -validation, contains("m1"))

h2o.data <- as.h2o(h2o.train)

y <- match('target', names(model.data.summary))

model <- 
  h2o.deeplearning(x = 1:(y-1),  # column numbers for predictors
                   y = y,   # column number for label
                   data = dat_h2o, # data in H2O format
                   classification = TRUE,
                   activation = "TanhWithDropout", # Tanh, Rectifier, TanhWithDropout, RectifierWithDropout
                   input_dropout_ratio = 0.25, # % of inputs dropout
                   hidden_dropout_ratios = c(0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                   train_samples_per_iteration = -2, #-2: autotuning
                   balance_classes = TRUE,
                   class_sampling_factors = c(.1,.9), #0, 1
                   #max_after_balance_size = .9,
                   fast_mode = TRUE,
                   #l1=c(0,1e-5),
                   shuffle_training_data = TRUE,
                   #rate = .005,
                   adaptive_rate = T,
                   rho = 0.99,
                   epsilon = 1e-8,
                   hidden = c(50,50,50,50,50), # 5 layers of 50 nodes
                   epochs = 100) # max. no. of epochs

h2o_p <- h2o.predict(model, as.h2o(localH2O, driver.summary))
h2o_p <- as.data.frame(h2o_p)

h2o.shutdown(prompt = FALSE)

