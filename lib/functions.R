###################################################################
### Memory-based Collaborative Filtering Algorithm ###
###################################################################

### Authors: Group 4
### Project 3
### ADS Spring 2018

####################################################
## Transform MS data
####################################################

MS_data_transform <- function(MS) {
  
  ## Calculate UI matrix for Microsoft data
  ## input: data   - Microsoft data in original form
  ## output: UI matrix
  
  
  # Find sorted lists of users and vroots
  users  <- sort(unique(MS$V2[MS$V1 == "C"]))
  vroots <- sort(unique(MS$V2[MS$V1 == "V"]))
  
  nu <- length(users)
  nv <- length(vroots)
  
  # Initiate the UI matrix
  UI            <- matrix(0, nrow = nu, ncol = nv)
  row.names(UI) <- users
  colnames(UI)  <- vroots
  
  user_locs <- which(MS$V1 == "C")
  
  # Cycle through the users and place 1's for the visited vroots.
  for (i in 1:nu) {
    name     <- MS$V2[user_locs[i]]
    this_row <- which(row.names(UI) == name)
    
    # Find the vroots
    if (i == nu) {
      v_names <- MS$V2[(user_locs[i] + 1):nrow(MS)]
    } else {
      v_names <- MS$V2[(user_locs[i] + 1):(user_locs[i+1] - 1)]
    }  
    
    # Place the 1's
    UI[this_row, colnames(UI) %in% v_names] <- 1
  }
  return(UI)
}

####################################################
## Transform movie_data
####################################################

movie_data_transform <- function(movie) {
  
  ## Calculate UI matrix for eachmovie data
  ## input: data   - movie data in original form
  ## output: UI matrix

  # Find sorted lists of users and vroots
  users  <- sort(unique(movie$User))
  movies <- sort(unique(movie$Movie))
  
  # Initiate the UI matrix
  UI            <- matrix(NA, nrow = length(users), ncol = length(movies))
  row.names(UI) <- users
  colnames(UI)  <- movies
  
  # We cycle through the users, finding the user's movies and ratings
  for (i in 1:length(users)) {
    user    <- users[i]
    movies  <- movie$Movie[movie$User == user]
    ratings <- movie$Score[movie$User == user]
    
    ord     <- order(movies)
    movies  <- movies[ord]
    ratings <- ratings[ord]
    
    # Note that this relies on the fact that things are ordered
    UI[i, colnames(UI) %in% movies] <- ratings
  }
  return(UI)
}  



####################################################
## Spearman, Pearson, Vector
####################################################

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

####################################################
## Implement Mean square difference
####################################################

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

####################################################
## Implement SimRank model 
####################################################

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

####################################################
## Prediction using similarity weights
####################################################
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


####################################################
## Ranking score error
####################################################

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

####################################################
## Mean absolute error (MAE)
####################################################

mae <-function(data_test, predictions_test) {
  
  n.items <- dim(data_test)[2]
  mae <- matrix(NA, nrow = 1, ncol = n.items)
  
  for (i in 1:n.items){
    what.to.predict<- which(!is.na(data_test[,i]))
    
    predictions <- predictions_test[what.to.predict]
    test.data <- data_test[what.to.predict]
    
    difference <- abs(predictions - test.data)
    mae[,i] <- sum(difference, na.rm = TRUE)/length(what.to.predict)
    
  }
    return(mae)
}


####################################################
## Neighborhood selection with Pearson correlation
####################################################

pred_topn <- function(data, simweights,n) {
  
  ## select top n neighbors
  ##
  ## input: data   - movie data or MS data in user-item matrix form
  ##        simweights - a matrix of similarity weights
  ##		n	-	top n neighbors ranked by strength of similarity
  ##
  ## output: prediction matrix
  
  # Initiate the prediction matrix.
  pred_mat <- data
  
  # Change MS entries from 0 to NA
  pred_mat[pred_mat == 0] <- NA
  
  #get the avg score across items for each user
  row_avgs <- apply(data, 1, mean, na.rm = TRUE)
  
  # Transform the UI matrix into a deviation matrix
  dev_mat     <- data - matrix(rep(row_avgs, ncol(data)), ncol = ncol(data))
  
  for(i in 1:nrow(data)) {
    
    # Find columns we need to predict for user i and sim weights for user i
    cols_to_predict <- which(is.na(pred_mat[i, ]))
    neighb_weights  <- simweights[i, ]
    
    #order the neighbors based on strength of correlation with user i:
    ord<-order(abs(neighb_weights),decreasing=TRUE)
    
    #weights for the top n neighbors
    neighb_weight_select<-head(neighb_weights[ord],n) 
    
    
    
    # weighted averages of the deviations
    weight_mat  <- matrix(rep(neighb_weight_select, ncol(data)), ncol = ncol(data))
    
    weight_sub <- weight_mat[, cols_to_predict]
    dev_sub    <- dev_mat[ ,cols_to_predict]
    
    #deviations for select neighbors
    neighb_dev_select<-head(dev_sub[ord,],n)
    
    pred_mat[i, cols_to_predict] <- row_avgs[i] +  apply(neighb_dev_select * weight_sub, 2, sum, na.rm = TRUE)/sum(neighb_weight_select, na.rm = TRUE)
    
  }
  
  return(pred_mat)
}



pred_threshold <- function(data, simweights, th) {
  
  ## select neighbors whose correlation with user i is above the given threshold
  ##
  ## input: data   - movie data or MS data in user-item matrix form
  ##        simweights - a matrix of similarity weights
  ##		th	-	threshold for similarity strength
  ##
  ## output: prediction matrix
  
  # Initiate the prediction matrix.
  pred_mat <- data
  
  # Change MS entries from 0 to NA
  pred_mat[pred_mat == 0] <- NA
  
  row_avgs <- apply(data, 1, mean, na.rm = TRUE)
  
  # Transform the UI matrix into a deviation matrix since we want to calculate
  dev_mat     <- data - matrix(rep(row_avgs, ncol(data)), ncol = ncol(data))
  
  for(i in 1:nrow(data)) {
    
    # Find columns we need to predict for user i and sim weights for user i
    cols_to_predict <- which(is.na(pred_mat[i, ]))
    neighb_weights  <- simweights[i, ]
    
    #select neighbors whose similarity score is above the threshold
    select_neighb<-which(abs(neighb_weights)>=th)
    
    
    #weights for the selected neighbors
    neighb_weight_select<-neighb_weights[select_neighb]
    
    
    
    # weighted averages of the deviations
    weight_mat  <- matrix(rep(neighb_weight_select, ncol(data)), ncol = ncol(data))
    
    weight_sub <- weight_mat[, cols_to_predict]
    dev_sub    <- dev_mat[ ,cols_to_predict]
    
    #deviations for select neighbors
    neighb_dev_select<-dev_sub[select_neighb,]
    
    pred_mat[i, cols_to_predict] <- row_avgs[i] +  apply(neighb_dev_select * weight_sub, 2, sum, na.rm = TRUE)/sum(neighb_weight_select, na.rm = TRUE)
    
  }
  
  return(pred_mat)
}



####################################################
## EM Algorithm
####################################################

############## Install library(matrixStats) ##############
library(matrixStats)

EM_train <- function(data, k, C = 12, tau = 0.01, ITER = 100){ 
  
  ############## Step 1 - Initial Conditions and Set up ##############
  n.users <- dim(data)[1]
  n.items <- dim(data)[2]
  
  mu <- rep(1/C, C) # vector length C 
  
  gamma <- array(1/k, dim = c(k, n.items, C)) # dims = ratings, items, clusters
  for (x in 1:C) {
    for (j in 1:n.items) {
      gamma[,j,x] <- (z<-runif(6))/sum(z)
    }
  }
  
  iter <- 1
  is.conv <- 100
  aic_pre <- 0
  aic.numer <- matrix(NA, nrow = n.users, ncol = C)
  aic <- matrix(NA, nrow = n.users, ncol = C) # cluster assignment/probability matrix, dims = users, clusters
  
  while((iter < ITER) | (is.conv > tau)) { # is.conv > tau
    
    ############## Step 2 - Expectation ##############
    ## Expectation Step
    gamma <- gamma/mean(gamma) # normalize gamma to avoid underflow 
    for(i in 1:n.users){
      items <-  which(!is.na(data[i, ]))# items that user i ranked/visited
      rank <- na.omit(data[i, ]) # specific rankings of items 
      
      for(c in 1:C) {      
        for(x in seq_along(rank)){ 
          aic.numer[i, c] <- mu[c]*gamma[rank[[x]], items[[x]], c] # extract the gamma that corresponds to the actual rating {r_i,j}
        }
      }
      
      aic[i, ] <- aic.numer[i, ]/sum(aic.numer[i, ]) # for each i, the denominator should be the same. 
      # return(aic)
    }
    
    
    ############## Step 3 - Maximization ##############
    ## Estimate Mu 
    mu <- apply(aic, 2, sum)/n.users # vector of cluster shares/probabilities
    
    ## Estimate Gamma
    indicator.numer <- array(0, dim = c(n.users,n.items, k))
    indicator.denom <- matrix(0, nrow = n.users, ncol = n.items)
    
    for (i in 1:k) 
    {
      indicator.numer[,,i][which(data == i)] <- 1 # indicator array to identify each instance in which user i rated item j with rating k
      indicator.numer <- aperm(indicator.numer, c(2,1,3)) # transposing so that the matrix is conformable to aic. new dims = items, users, ratings
    
      numer <- apply(indicator.numer, 3, function(x) x%*%aic) # each element of resulting array represents the sum of the class C aic weights of every user who rated that jth item with that rating k
      numerator <- array(numer, c(n.items, C, k)) # puts data in the correct shape. dims = (items, clusters, ratings) 
    
    
     indicator.denom[which(data != 0)] <- 1 # indicator array to identify each instance in which user i rated movie j with any rating
     indicator.denom <- t(indicator.denom) # dims = movies, users. 
     denominator <- indicator.denom%*%aic # each element of the resulting matrix represents the sum of the class C aic weights of every user who rated movie j with any rating. dims = (items, clusters). 
     # Note: denominator is common across ratings k, but not across clusters, there for the following code would be unecessary: denominator <- apply(indicator.denom, 3, function(x) x%*%aic)
    
    #result <- nominator/denominator # dims = movies, clusters, ratings
      result <- apply(numerator, 3, function(x) x/denominator) # divides each numerator element by its respective denominator value. 
      gamma <- array(result, c(n.items, C, k)) # dims = (items, clusters, ratings)
      gamma <- aperm(gamma, c(3, 1, 2)) # dims = ratings, items, clusters
      
    
    ############## Step 4 - Convergence Conditions using l-2 norm ##############
      is.conv <- norm(aic - aic_pre, type = "2") # measure of the change in cluster assignments to determine when to terminate the algorithm.
    
      aic_pre <- aic
      iter <- iter + 1
    }
  }
  ############## Step 5 - Hard Assignment ##############
  hard.assignments <- (aic == rowMaxs(aic)) + 0 # a function from "MatrixsStats" library
  rownames(hard.assignments) <- rownames(data)
  colnames(gamma) <- colnames(data)
  
  return(list("hard.assignments" = hard.assignments, "aic" = aic, "final.conv" = is.conv, "aic_pre" = aic_pre, "gamma" = gamma, "mu" = mu, "iterations" = iter))
}



####################################################
## Bayesian Prediction
####################################################


bayesian_predict <-function(active.users, hard.assignments, gamma, k) {
  n.users <- dim(active.users)[1]
  n.items <- dim(active.users)[2]
  C <- ncol(hard.assignments)
  
  users.subset <- rownames(active.users)
  items.subset <- colnames(active.users)
  hard.assignments <- hard.assignments[rownames(hard.assignments) %in% users.subset,]
  gamma <- gamma[,colnames(gamma) %in% items.subset,]
  
  
  weighted.rating <- apply(gamma, 3, function(x) {return(x*(1:k))}) 
  weighted.rating <- array(weighted.rating, c(k, n.items, C)) # these two lines multiply the probability of r_{k,j,c} by the value of k for all k,j,c, resulted in a weighted rating. dims = k, j, c
  
  expected.rating <- t(colSums(weighted.rating)) # this sums the weighted ratings for each j,c pair i.e. weighted average of ratings for each j,c pair. this is the expected rating for each cluster for each movie. dims = (C, j)
  
  predicted.ratings <- hard.assignments %*% expected.rating # picks the expected rating for user i on movie j based on user i's hard cluster assignment, for all i,j. dims = i, j
  
  predicted.ratings
  return(predicted.ratings)
}

