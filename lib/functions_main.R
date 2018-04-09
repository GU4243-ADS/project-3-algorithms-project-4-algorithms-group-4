# Functions from Memory-Based

# Functions from Neighboring


EM_train <- function(data, k, C = 5, tau = 0.01, ITER = 1000){ 
  library(matrixStats) # call library for matrix algebra
  
  n.users <- dim(data)[1]
  n.items <- dim(data)[2]
  
  mu <- rep(1/C, nrow = C)
  gamma <- array(1/6, dim = c(k, n.items, C)) # dims = ratings, items, and clusters
  
  iter <- 1
  
  aic.numer <- matrix(NA, nrow = n.users, ncol = C)
  aic <- matrix(NA, nrow = n.users, ncol = C) 
  
  gamma <- gamma/mean(gamma) # normalize gamma to avoid underflow 
  
  while(is.conv > tau | iter < ITER) {
    
    #### E-STEP
    for(i in 1:n.users){
      items <- which(!is.na(data[i, ])) # items that user i ranked/visited
      rank <- na.omit(data[i, ]) # specific rankings of items
      
      for(c in 1:C){
        for(x in 1:length(items)){
          aic.numer[i, c] <- mu[c]*gamma[rank[[x]], items[[x]], c] # extract the gamma that
          # corresponds to the actual rating {r_i,j}
        }
      }
      
      aic[i, ] <- aic.numer[i, ]/sum(aic.numer[i, ]) 
      # for each i, the denominator should be the same
      return(aic)
    }
    
    #### M-STEP
    
    ## Estimate Mu
    mu <- apply(aic, 2, sum)/n.users # vector of cluster shares/probabilities
    
    ## Estimate Gamma
    
    indicator.numer <- array(0, dim = c(n.users, n.items, k))
    indicator.denom <- matrix(0, nrow = n.users, ncol = n.items)
    
    for(i in 1:k){
      indicator.numer[,,i][which(data == i)] <- 1 # indicator array to identify each 
        # instance in which user i rated item j with rating k
      
    }
    indicator.numer <- aperm(indicator.numer, c(2, 1, 3)) # transposing so that the 
      # matrix is conformable to aic. new dims = items, users, ratings
    numer <- apply(indicator.numer, 3, function(x) x%*%aic) # each element of resulting 
      # array represents the sum of the class C aic weights of every user who rated that 
      # jth item with that rating k
    numerator <- array(numer, c(n.items, C, k)) # puts data in the correct shape
    
    indicator.denom[which(data != 0)] <- # indicator array to identify each instance in
      # which user i rated item j with any rating
    indicator.denom <- t(indicator.denom) # dims = items, users
    denominator <- indicator.denom%*%aic # each element of the resulting matrix represents 
      # the sum of the class C aic weights of every user who rated item j with any rating
      # dims = (item, clusters)
    
    result <- apply(numerator, 3, function(x) x/denominator) # divides each numerator
      # element by its respective denominator value
    gamma <- array(result, c(n.items, C, k)) # dims = items, clusters, ratings
    gamma <- aperm(gamma, c(3, 1, 2)) # dims = ratings, items, and clusters (match above)
    
    ### Convergence Check
    is.conv <- norm(aic - aic.pre, type = "2") # measure the change in cluster assignments
      # to determine when to terminate algorithm 
    
    aic_pre <- aic
    iter = iter + 1
  }
    
    ### Hard Assignment
    hard.assignments <- (aic == rowMaxs(aic)) + 0 # create hard assignments
    
    return(list("hard.assignments" = hard.assignments, "gamma" = gamma, "mu" = mu))
}




# Functions for predicting