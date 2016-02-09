auc <- function(outcome, proba){
  
  outcome <- as.vector(outcome)
  proba <- as.vector(proba)
  
  N <- length(proba)
  N_pos <- sum(outcome)
 
  df <- data.frame(out = outcome, prob = proba)
  df <- df[order(-df$prob),]
  df$above <- (1:N) - cumsum(df$out)
  return( 1- sum( df$above * df$out ) / (N_pos * (N-N_pos) ) )
}

##function for 
auc.gbm <- function(actual, dtrain) {
  
  preds <- as.vector(getinfo(dtrain, "label"))
  outcome <- as.vector(actual)
  
  N <- length(preds)
  N_pos <- sum(outcome)
 
  df <- data.frame(out = outcome, prob = preds)
  df <- df[order(-df$prob),]
  df$above <- (1:N) - cumsum(df$out)
  auc <- ( 1- sum( df$above * df$out ) / (N_pos * (N-N_pos) ) )
  
  return(list(metric = "AUC", value = auc))
}