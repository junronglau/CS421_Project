The following writeups are a summarized version of the project submission for CS421 Introduction to Machine Learning. Working alongside me are my team members Ambrose Tan, Ong De Lin, Josh Lim, Janell Lee. 

# Objective
We aim to improve the representation and prediction of minority classes in text classification via ensemble methods for imbalanced datasets. 

# Motivation

Imbalanced dataset can negatively affect classification performance. The classifier trained on imbalanced data may be biased towards the majority class. Often, real-world data sets are composed of an imbalance in class, which introduces implications and complexities in solving problems. In our project, we introduced a novel approach inspired by numerous research literature to better represent minority classes in the context of text classification. 

# Dataset
Retrieved from: https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set 
The review dataset is a set of 53,000 reviews from IMDB, Yelp and Amazon (comprising of a set of words) for movies, service providers and products, accompanied with a binary classification where 0 indicates a negative sentiment for the review and 1 a positive sentiment. We artifically downsampled the positive class to introduce a class imbalance.


# Proposed approach

We propose a combinational approach of Clustering and and then an ensemble method of Underbagging, synthesized from the literature reviews we reviewed, to be applied on text classification. We represented our documents with the Siamese Bag of Words (https://arxiv.org/abs/1606.04640) approach. The base classifier that we will be using to validate our model is Linear Support Vector Machines as it showed the highest promise during our preliminary testing. The flow of the process can be seen below.

<img src="https://imgbbb.com/images/2019/12/08/Methodflow.png" alt="Methodflow.png" border="0" />

* Cluster the majority class together so documents of similar meanings will be in the same cluster
* To form our bootstrap samples, we randomly selected instances from each cluster so the ratio of majority class matches the size of our minority class. This will then form one bootstramp sample, and the process is repeated for N bootstrap samples.
* Each bootstrap sample is trained on linear SVM to classify the sentiments. The results are then aggregated via a soft-voting approach to consider the certainty each model is in their prediction. 

# Results and discussion

To measure the validity and a comprehensive evaluation of our model, we have chosen the following metrics:

* Precision score: 0.96
* Recall score: 0.76
* F1 score: 0.85
* MCC score: 0.33
* AUC score: 0.83

<a href="https://imgbbb.com/image/LSy9Gn"><img src="https://imgbbb.com/images/2019/12/08/download.png" alt="download.png" border="0" /></a>

In comparing our approach with existing methods such as random oversampling, the performance of our model is significantly dependent on the quality of the trained embeddings because we will be clustering then sampling from it. Hence, embeddings which are not sufficiently trained will impact the quality of our clusters which leads to worse resampling results, and ultimately affecting our final performance. 

# Limitations

Currently, we are sampling with replacements and are not fully utilizing all data points where some may be important for classification. Although word embeddings do not utilize as much memory as using Bag of Words or TF-IDF representation, it might still be computationally expensive as we must first train our own word embedding models on each of the original datasets. We have the option to use pre-trained word embedding models, but it will be highly dependent on the context of the data it was trained in. In training our own model, we also need a large amount of data in order for the embeddings to be meaningful.

Another limitation of our ensemble sampling method that might happen is when the dataset is highly imbalanced. For example, if there exist only 10 minority classes but 10,000 majority classes, we would only be training our base classifiers with bootstrap samples of size 20. Hence, the results obtained may not be useful and meaningful, and would be outperformed by other traditional resampling methods.


