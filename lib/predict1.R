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
weights_MSD_train_MS <- read.csv(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/weights_MSD_train_MS.csv")
names(weights_MSD_train_MS) <- NULL
weights_MSD_train_MS <- weights_MSD_train_MS[,-1]
pred_ms <- pred_matrix(MS_train,weights_MSD_train_MS)

is.numeric(MS_train)
is.numeric(weights_MSD_train_MS)

weights_MSD_train_MS <- as.numeric(unlist(weights_MSD_train_MS))

is.numeric(weights_MSD_train_MS)
weights_MSD_train_MS <- as.matrix(weights_MSD_train_MS)
is.numeric(weights_MSD_train_MS)
is.numeric(MS_train)

pred_web_meansquare <- pred_matrix(MS_UI, weights_MSD_train_MS)
save(pred_web_meansquare, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_web_meansquare.Rdata")
readRDS(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/web_spearman.Rdata")
pred_web_spearman <- pred_matrix(MS_UI, web_spearman)
save(pred_web_spearman, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_web_spearman.Rdata")
readRDS(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_web_spearman.Rdata")
write.csv(pred_web_spearman, file='/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_web_spearman.csv', row.names = FALSE)
web_vector <- readRDS(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/web_vector.Rdata")
pred_web_vector <- pred_matrix(MS_UI, web_vector)
save(pred_web_vector, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_web_vector.Rdata")
write.csv(pred_web_vector, file='/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_web_vector.csv', row.names = FALSE)

movie_spearman <- readRDS(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/movie_spearman.Rdata")
pred_movie_spearman <- pred_matrix(movie_UI, movie_spearman)
save(pred_movie_spearman, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_spearman.Rdata")
write.csv(pred_movie_spearman, file='/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_spearman.csv', row.names = FALSE)

movie_MSD <- readRDS(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/weights_MSD_train_Movie.Rdata")
movie_MSD <- as.matrix(as.numeric(unlist(movie_MSD)))
pred_movie_MSD <- pred_matrix(movie_UI, movie_MSD)
save(pred_movie_MSD, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_MSD.Rdata")
write.csv(pred_movie_MSD, file='/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_MSD.csv', row.names = FALSE)


movie_vector <- readRDS(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/movie_vector.Rdata")
# movie_MSD <- as.matrix(as.numeric(unlist(movie_MSD)))
pred_movie_vector <- pred_matrix(movie_UI, movie_vector)
save(pred_movie_vector, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_vector.Rdata")
write.csv(pred_movie_vector, file='/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_vector.csv', row.names = FALSE)

load(file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/usersim.Rdata")
movie_simrank <- user_sim
# movie_MSD <- as.matrix(as.numeric(unlist(movie_MSD)))
pred_movie_simrank <- pred_matrix(movie_UI[1:1000,1:1000], movie_simrank)
save(pred_movie_simrank, file = "/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_simrank.Rdata")
write.csv(pred_movie_simrank, file='/Users/yunli/Documents/GitHub/project-3-algorithms-project-4-algorithms-group-4/output/pred_movie_simrank.csv', row.names = FALSE)



