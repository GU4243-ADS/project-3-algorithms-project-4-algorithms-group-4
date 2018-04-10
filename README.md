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
Using different similarity weights, the memory based model compares all users to all other users. It is essential to use different similarity weights and various measures of correlation compute different things. For example, Pearson correlation measures a linear relationship, whereas Spearman correlation is monotonic. 

##### Results
| | | Pearson | Spearman | Vector | Mean Square | Sim Rank | 
|:---:| :---:| :---:| :---:| :---:|:---:|---
Rank Score| MS | 37.57| 37.57| 37.82|45625.77|N/A
MAE|Movie|1.085|1.085|1.095|326.54 |1.0497
Run Time| MS| 0.76H| 0.70H|0.62H|2.7H|N/A
Run Time| Movie| 1.75H|2.28H|1.5H|1.5H (1000rows)|8.5H


#### Selecting Neighbors
After selecting a neighborhoods, we combine ratings to make a prediction. 
|	| Correlation Threshold | Top-N-Neighbors| Computational Time |

#### movie data 
________________________________________________________

| top n |	computational time |	threshold	| computational time |
|:---:| :---:| :---:| :---
10%|	 1963.017					| 0.9			| 525.02
30%|	 2303.331					| 0.7			| 1087.584
50%|	 2757.009					| 0.5			| 1244.405
70%| 3178.991					| 0.2			| 2296.728
					
					
#### MS data ____________________________________________________
|:---:| :---:| :---:| :---:
|top n | computational time | threshold | computational time |
10%	| 224.677				| 0.8		| 27.608
30%	| 275.706				| 0.6		| 77.137
50%	| 302.482				| 0.4		| 113.58
70%	| 350.525				| 0.2		| 212.585


### Model-based Algorithm

##### Description
The EM algorithm uses clusters to predict the ranks within a cluster. For example, can we group users who rate some movies similarily into a single cluster and predict the outcome of that cluster? Unlike the memory-based approach, the model-based approach is less computationally taxing as one only needs to store the cluster information rather than entire dataset. 

##### Results 
### Elapsed Time

     user    system   elapsed 
28452.009   875.216 29383.780 
---> 8.162161 hours

### Total iterations: 750

### Final tau (L-2 norm of difference): 0.0115
- We suspect the tau had flatlined for a long time

### Dispersion between clusters (percents):

 [1]  0.09891197  0.00000000  0.01978239 10.28684471  0.05934718
 [6]  0.00000000  0.53412463  0.00000000  0.03956479  0.00000000
[11] 88.96142433  0.00000000
- 88% in group 11 which is interesting because in 3 iterations we had 64% in cluster 8 
- that is more intuitive, it would be unlikely that all latent groupings were even sized

### distribution of Mu_c (percent chance):

[1]  8.325108  5.676666  8.031387 13.194455  8.716813  3.106978
 [7] 12.041968  3.526480 10.977759  5.100484 13.848172  7.453731
- This may mean that even though one group is bigger than the other, a given user has an equal chance of being in any group which mean that the true latent group 11 is just huge. 

### Test Error
MAE -> 1.111207

Further results and graphs will be in our in class presentation.


**Contribution statement**: Mingyue and Yun completed the memory model and wrote the ranking score evaluation. More specifcally, Yun developed Mean-square similarity and Simrank while Mingyue developed Pearson Correlation, Spearman Correlation and Vector Similarity. Mingyue and Yun generated predictions and evaluations of these five similarity weights. Judy did the neighborhood selection model on Pearson correlations. Noah and Nicole worked on the EM algorithm. More specifically, Nicole and Noah tag-teamed the E-step and the Bayesian prediction function. Noah wrote the M-step and Nicole worked on the mean absolute error. Nicole annotated the main.Rmd and the readme files on Github. Yun made the powerpoint presentation. All team members contributed to the GitHub repository and prepared the presentation. All team members approve our work presented in our GitHub repository including this contribution statement.

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
