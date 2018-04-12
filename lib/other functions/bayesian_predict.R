
bayesian_predict <-function(labeled.data, hard.assignments, gamma) {
  n.users <- dim(labeled.data)[1]
  n.items <- dim(labeled.data)[2]
  C <- ncol(hard.assignments)
  k <- (length(unique(as.vector(labeled.data))) - 1)
  
  ## subset data so we predict only on to the positions and dimensions of the labeled test data matrix
  users.subset <- rownames(labeled.data)
  items.subset <- colnames(labeled.data)
  hard.assignments <- hard.assignments[rownames(hard.assignments) %in% users.subset,]
  gamma <- gamma[, colnames(gamma) %in% items.subset, ] # dims = ratings, items, clusters # for k=1 this becomes a matrix, dims = items, clusters
  gamma <- array(gamma, dim = c(k, n.items, C)) # redundant for k>1, but this allows the function to work on k=1 MS data. Each row from above is put into an array col.
  
  ## Make predictions
  weighted.rating <- apply(gamma, 3, function(x) {return(x*(1:k))}) #  apply() returns a weirdly shaped matrix again which must be corrected in the next step.
  weighted.rating <- array(weighted.rating, c(k, n.items, C)) # these two lines multiply the probability of r_{k,j,c} by the value of k for all k,j,c, resulted in a weighted rating. dims = k, j, c
  
  expected.rating <- t(colSums(weighted.rating)) # this sums the weighted ratings for each j,c pair i.e. weighted average of ratings for each j,c pair. this is the expected rating for each cluster for each movie. dims = (C, j)
  predicted.ratings <- hard.assignments %*% expected.rating # picks the expected rating for user i on movie j based on user i's hard cluster assignment, for all i,j. dims = i, j
  
  #rownames(predicted.ratings) <- rownames(labeled.data)
  #colnames(predicted.ratings) <- colnames(labeled.data)
  
  predicted.ratings
}