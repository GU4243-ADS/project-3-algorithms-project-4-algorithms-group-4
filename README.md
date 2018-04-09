# Spring2018


# Project 3: Algorithm Implementation and Evaluation

----


### [Project Description](doc/)

Term: Spring 2018

+ Project title: Collaborative Filtering
+ Team Number: Group 4
+ Team Members: Noah Chasek-Macfoy, Judy Jinhui Cheng, Mingyue Kong, Yun Li, Nicole Smith
+ Project summary: In our daily lives, we engage with ads, products, music, movies, and other information that are based on recommendation systems. Using memory-based and model-based algorithms, we conducted collaborative filtering on two datasets, one on website data (Microsoft Web data) with range 0 to 1 and another on movie ranking data with (EachMovie data) range from 1 to 6. The memory-based model considered different similarity weights including Pearsons's correlation, Spearman's correlation, entropy, mean squared difference, and SimRank. From the memory based similarity weights, we used a neighbor selecting process (correlation threshold, and top-n-neighbors) in an attempt to enhance our predictions. We also engaged with the EM algorithm for clustering.  


#### Memory-based Algorithm 

##### Description

##### Results
| | | Pearson | Spearman | Vector | Mean Square | Sim Rank | 
|:---:| :---:| :---:| :---:| :---:|:---:|---
Rank Score| MS | 37.57| 37.57| 37.82|45625.77| 
MAE|Movie|1.19|1.19| | | 
Run Time| MS| 0.76H| 0.70H|0.62H|2.7H| 
Run Time| Movie| 1.75H|2.28H|1.5H| | (1000 rows)


##### Selecting Neighbors
|	| Correlation Threshold | Top-N-Neighbors| Computational Time |
--- | :---: |	:---: | ---
Neighbors (Pearson)| 	|	|


#### Model-based Algorithm

##### Description

##### Results 
|	| Test Error  | Computational Time |
 ---|:---: | ---
| Movie | (MAE)	| |
| website | (Rank score)| 13.23 minutes| 


Contribution statement: Mingyue and Yun completed the memory model and wrote the ranking score evaluation. More spefically, Yun developed Mean-square similarity and Simrank while Mingyue developed Pearson Correlation, Spearman Correlation and Vector Similarity. Mingyue and Yun generated predictions and evaluetions of these five similarity weights. Judy did the neighborhood selection model on Pearson correlations. Noah and Nicole worked on the EM algorithm. More spefically, Nicole and Noah tag-teamed the E-step and the Bayesian prediction function. Noah wrote the M-step and Nicole worked on the mean absolute error. Nicole annotated the main.Rmd and the readme files on Github. Yun made the powerpoint presentation. All team members contributed to the GitHub repository and prepared the presentation. All team members approve our work presented in our GitHub repository including this contribution statement.

#### References 

- Breese, J. S., Heckerman, D., & Kadie, C. (1998, July). Empirical analysis of predictive algorithms for collaborative filtering. In Proceedings of the Fourteenth conference on Uncertainty in artificial intelligence (pp. 43-52). Morgan Kaufmann Publishers Inc..

- Herlocker, J. L., Konstan, J. A., Borchers, A., & Riedl, J. (1999, August). An algorithmic framework for performing collaborative filtering. In Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval (pp. 230-237). ACM.

- Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in artificial intelligence, 2009, 4.

- Jeh, G., & Widom, J. (2002, July). SimRank: a measure of structural-context similarity. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 538-543). ACM.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── data/
├── docs/
├── figs/
├── lib/
└── output/
```

Please see each subfolder for a README file.
