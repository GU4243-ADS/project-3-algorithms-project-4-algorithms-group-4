# Functions from Memory-Based

# Implement Mean square difference

MS_train <- MS_UI[1:4151,]
Movie_train <- movie_UI[1:1000,]

train_start <- Sys.time()
meanSquareDiff <- function(df){
  m <- dim(df)[1] #user number
  dissim <- matrix(rep(NA, m * m), m, m)
  dissim <- data.frame(dissim)
  users <- rownames(df)
  colnames(dissim) <- users
  rownames(dissim) <- users
  for (i in 1:m){
    for (j in 1:m){
      r_i <- df[i,]
      r_j <- df[j,]
      dissim[i, j] <- mean((r_i - r_j)^2, na.rm = T)
    }
  }
  maxDissim <- max(dissim)
  sim <- (maxDissim - dissim)/maxDissim
  return (sim)
  
}

weights_MSD_train_MS <- meanSquareDiff(MS_train)
saveRDS(weights_MSD_train_MS, file = "../output/weights_MSD_train_MS.RData")
write.csv(weights_MSD_train_MS, file='../output/weights_MSD_train_MS.csv', row.names = TRUE)

weights_MSD_train_Movie <- meanSquareDiff(Movie_train)
saveRDS(weights_MSD_train_Movie, file = "../output/weights_MSD_train_Movie.RData")
write.csv(weights_MSD_train_Movie, file='../output/weights_MSD_train_Movie.csv', row.names = TRUE)


train_end <- Sys.time()
train_time = train_end - train_start

#Implement SimRank model

#Control function
#Read raw data files
read.movie.data = T

#Reshape data to wide format
reshape.movie = T

#Load Rdata
load.movie.data = T

#Implementation algorythims
model.sim.rank <- T

#Read Movie Dataset
if (read.movie.data){
  
  #Read dataset from directory
  movie.data.train <- read.csv("../data/eachmovie_sample/data_train.csv")
  movie.data.test <- read.csv("../data/eachmovie_sample/data_test.csv")
  # movie.data.train <- movie.data.train[1:1855,]
  
  movie.data.train <- movie.data.train[,-1]
  movie.data.test <- movie.data.test[,-1]
  
  #Check dataset information
  paste("Data Dimension:", dim(movie.data.train)[1], "x", dim(movie.data.train)[2])
  paste("Number of Unique Movies:",length(unique(movie.data.train[,1])))
  paste("Number of Unique Users:",length(unique(movie.data.train[,2])))
  paste("Number of 0 ratings:", sum(movie.data.train[,3] == 1))
  
}

#Data Processing
if(reshape.movie){
  
  #Reshapes dataframe
  movie.data.train <- reshape(movie.data.train, 
                              v.names = "Score", 
                              direction = "wide", 
                              idvar = "User", 
                              timevar = "Movie")
  
  movie.data.test <- reshape(movie.data.test, 
                             v.names = "Score", 
                             direction = "wide", 
                             idvar = "User", 
                             timevar = "Movie") 
  
  
  
  paste("Data Dimension:", dim(movie.data.train)[1], "x", dim(movie.data.train)[2])
  paste("Data Dimension:", dim(movie.data.test)[1], "x", dim(movie.data.test)[2])
  
  #Save files to data directory
  save(movie.data.train, file = "../data/movie_data_train_wide.Rdata")
  save(movie.data.test, file = "../data/movie_data_test_wide.Rdata")
}

#Load Rdata files
if(load.movie.data){
  load("../data/movie_data_train_wide.Rdata")
  load("../data/movie_data_test_wide.Rdata")
}

#Helper functions for SimRank Model

# returns the corresponding row or column for a user or movie.
get_movies_num <- function(user){
  u_i <- match(user, users)
  return(graph[u_i,-1])
}

get_users_num <- function(movie){
  m_j <- match(movie, movies)
  return(graph[,m_j+1])
}

# return the users or movies with a non zero
get_movies <- function(user){
  series = get_movies_num(user)
  return(movies[which(series!=0)])
}

get_users <- function(movie){
  series = get_users_num(movie)
  return(users[which(series!=0)])
}

user_simrank <- function(u1, u2, C) {
  if (u1 == u2){
    return(1)
  } else {
    pre = C / (sum(get_movies_num(u1)) * sum(get_movies_num(u2)))
    post = 0
    for (m_i in get_movies(u1)){
      for (m_j in get_movies(u2)){
        i <- match(m_i, movies)
        j <- match(m_j, movies)
        post <- post + movie_sim[i, j]
      }
    }
    return(pre*post)
  }
}

movie_simrank <- function(m1, m2, C) {
  if (m1 == m2){
    return(1)
  } else {
    pre = C / (sum(get_users_num(m1)) * sum(get_users_num(m2)))
    post = 0
    for (u_i in get_users(m1)){
      for (u_j in get_users(m2)){
        i <- match(u_i, users)
        j <- match(u_j, users)
        post <- post + user_sim[i, j]
      }
    }
    return(pre*post)
  }
}

simrank <- function(C=0.8, times = 1, calc_user = T, calc_movie = F, data){
  
  for (run in 1:times){
    
    if(calc_user){
      for (ui in users){
        for (uj in users){
          i = match(ui, users)
          j = match(uj, users)
          user_sim[i, j] <<- user_simrank(ui, uj, C)
        }
      }
    }
    if(calc_movie){
      for (mi in movies){
        for (mj in movies){
          i = match(mi, movies)
          j = match(mj, movies)
          movie_sim[i, j] <<- movie_simrank(mi, mj, C)
        }
      }
    }
  }
}

if(model.sim.rank){
  graph <- movie.data.train[1:1000, ]#[1:5055, 1:1620]
  
  graph[is.na(graph)] <- 0
  
  graph[,-1][graph[,-1] < 5] <- 0
  graph[,-1][graph[,-1] >= 5] <- 1
  
  
  # set similarity matrices to be calculated
  calc_user = T
  calc_movie = F
  
  # initialize the similarity matrices
  user_sim <- diag(dim(graph)[1])
  movie_sim <- diag(dim(graph)[2])
  
  # create list of users and movies
  users <- graph[,1]
  movies <- colnames(graph[,-1])
  
  simrank(0.8, 1)
  
  colnames(user_sim) <- users
  user_sim <- cbind(users, user_sim)
  write.csv(user_sim, file='../output/usersim.csv', row.names = FALSE)
  #Save files to data directory
  save(weights_usersim_train_Movie, file = "../output/weights_usersim_train_Movie.Rdata")
  
}

#Spearman+pearson+vector

all_weight <- function(data,method){
  library(lsa)
  data <- as.matrix(data)
  weight_mat <- matrix(NA,nrow=nrow(data),ncol=nrow(data))
  for(i in 1:nrow(data)){
    weight_mat[i,] <- apply(data,1,function(x){
      index <- (!is.na(x))&(!is.na(data[i,]))
      if(sum(index)==0){
        return(0)
      }else{
        if(method == 'pearson'){
          return(cor(data[i,index],x[index],method='pearson'))
        }else if(method == 'spearman'){
          return(cor(data[i,index],x[index],method='spearman'))
        }else if(method == 'entropy'){
          library(infoheo)
          return(mutinformation(data[i,index],x[index],method='emp'))
        }else if(method == 'vector'){
          return(cosine(data[i,index],x[index]))
        }
      }
    })
  }
  return(round(weight_mat,4))
}

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
pred_matrix <- function(data, simweights) {
  
  ## Calculate prediction matrix
  ##
  ## input: data   - movie data or MS data in user-item matrix form
  ##        simweights - a matrix of similarity weights
  ##
  ## output: prediction matrix
  
  # Initiate the prediction matrix.
  pred_mat <- data
  
  # Change MS entries from 0 to NA
  pred_mat[pred_mat == 0] <- NA
  
  row_avgs <- apply(data, 1, mean, na.rm = TRUE)
  
  for(i in 1:nrow(data)) {
    
    # Find columns we need to predict for user i and sim weights for user i
    cols_to_predict <- which(is.na(pred_mat[i, ]))
    num_cols        <- length(cols_to_predict)
    neighb_weights  <- simweights[i, ]
    
    # Transform the UI matrix into a deviation matrix since we want to calculate
    # weighted averages of the deviations
    dev_mat     <- data - matrix(rep(row_avgs, ncol(data)), ncol = ncol(data))
    weight_mat  <- matrix(rep(neighb_weights, ncol(data)), ncol = ncol(data))
    
    weight_sub <- weight_mat[, cols_to_predict]
    dev_sub    <- dev_mat[ ,cols_to_predict]
    
    pred_mat[i, cols_to_predict] <- row_avgs[i] +  apply(dev_sub * weight_sub, 2, sum, na.rm = TRUE)/sum(neighb_weights, na.rm = TRUE)
    print(i)
  }
  
  return(pred_mat)
}


# Rank score
rank_scoring <- function(predicted, web_mini_test, alpha){
  
  visited_ind <- apply(web_mini_test, 1, function(rrr){return(which(rrr==1))})
  ord <- t(apply(predicted, 1, function(rrr){return(order(rrr,decreasing = T))})) 
  R_a_s <- rep(NA, nrow(web_mini_test))
  R_a_max <- rep(NA, nrow(web_mini_test))
  
  for(a in 1:nrow(web_mini_test)){
    d<-mean(predicted[a,])
    j<-ord[a,] 
    m<-ifelse((predicted[a,]-d)>0,(predicted[a,]-d),0)
    
    R_a_s[a] <- sum( m / 2^((j-1)/(alpha-1)) )
    R_a_max[a] <- length(visited_ind[[a]])
  }
  
  
  R <- sum(R_a_s) / sum(R_a_max)*100
  return(R)
}

