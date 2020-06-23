# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:54:20 2017

@author: Farshid Rayhan, United International University

"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier
from cus_sampling import cus_sampler


class CUSBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(replacement=False)

        ## Some other samplers to play with ######
        # self.undersampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

    def fit(self, X_train, y_train, number_of_clusters, percentage_to_choose_from_each_cluster):
        self.models = []
        self.alphas = []

        N, _ = X_train.shape
        # ,_ 는 특정 값을 무시하기 위해 사용됌ex. x, _ ,y = (1, 2, 3)  ..x=1, y=3
        W = np.ones(N) / N
        
        # W = weight

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')


            X_undersampled, y_undersampled, chosen_indices = cus_sampler(X_train,y_train, number_of_clusters, 
                                                                         percentage_to_choose_from_each_cluster)


            tree.fit(X_undersampled, y_undersampled,
                   sample_weight=W[chosen_indices])  #fitting tree with cluster-sampled instances

            P = tree.predict(X_train) #predicting the trained tree with X_train instances, which is not the undersampled instances
            P_int = P.astype(int) #for indexing
            
            Prediction = np.ones(N)  #to index negative values
            negative_index = (Prediction != P_int) #indexes of prediction 0
            Prediction[negative_index] = -1
            
            y_train_value = np.ones(N) #to indexnegative values                         
            y_int = y_train.astype(int)
            negative_index_y = (y_train_value != y_int) #index of prediction (label as 0) to -1
            
            y_train_value[negative_index_y] = -1
            
            err = np.sum(W[P != y_train])
            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err)) #alpha is assigned vote based on error
                    W = W * np.exp(-alpha * y_train_value * Prediction)  # if y_train_value & Prediction is same, weight become lower
                    W = W / W.sum()  # normalize so it sums to 1         # if y_train_value & Prediction is not same, weight become large.
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)

    def predict(self, X_test):
        N, _ = X_test.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            Prediction = np.ones(N) 
            P = tree.predict(X_test)
            P_int = P.astype(int)
            
            negative_index = (Prediction != P_int)
            
            Prediction[negative_index] = -1 # instance that have been classified as negative(0) becomes -1
            
            FX += alpha * Prediction # alpha1 * 1 + alphas2 *-1 +...
        FX = np.sign(FX)         #threshold = 0
        
        FX[FX==-1] = 0   
            
        
        return FX #0 for negative, 1 for positive

    def predict_proba(self, X_test):
        # if self.alphas == 'SAMME'
        #재호 수정 5/20
        
        N, _ = X_test.shape
        proba = np.zeros([N,2])
        for tree, alpha in zip(self.models, self.alphas):
            each_prob = tree.predict_proba(X_test)*alpha
            proba += each_prob
        
        normalizer = sum(self.alphas)
        if normalizer <= 0:
            normalizer = 0.000001
        
        proba = proba / normalizer
     
        #proba = sum(tree.predict_proba(X_test) * alpha for tree , alpha in zip(self.models,self.alphas) )


        #proba = np.array(proba)


        #proba = proba / sum(self.alphas)

        #proba = np.exp((1. / (2 - 1)) * proba)
        #normalizer = proba.sum(axis=1)[:, np.newaxis]
        #normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        #proba = proba /  normalizer

        # print(proba)
        return proba
