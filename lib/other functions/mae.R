load("../data/movie_test.RData")
load("../data/samplepred.rdata")

data_test <- movie_UI_test
predictions_test <- test

mae <-function(data_test, predictions_test) {
  
  n.items <- dim(data_test)[2]
  mae <- matrix(NA, nrow = 1, ncol = n.items)
  
  for (i in 1:n.items){
    what.to.predict<- which(!is.na(data_test[,i]))
    
    predictions <- predictions_test[what.to.predict]
    test.data <- data_test[what.to.predict]
    
    difference <- abs(predictions - test.data)
    mae[,i] <- sum(difference, na.rm = TRUE)/length(what.to.predict)
    
    return(mae)
  }
}