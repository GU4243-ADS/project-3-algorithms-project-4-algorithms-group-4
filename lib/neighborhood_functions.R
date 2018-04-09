###################################################################
### NEIGHBORHOOD SELECTION WITH PEARSON CORRELATION ###
###################################################################

### Author: Judy Cheng
### Project 3--group 4
### ADS Spring 2018



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

