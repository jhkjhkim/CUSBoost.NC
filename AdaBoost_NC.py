
"""
# -*- coding: utf-8 -*-

Created on Tue May 19 14:49:24 2020

@author: Jaeho

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


class AdaboostNC_Classifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(replacement=False)

        ## Some other samplers to play with ######
        # self.undersampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

    def fit(self, X_train, y_train, lambda_):
        self.models = []
        self.alphas = []

        

        N, _ = X_train.shape
        
        W = np.ones(N) / N # W = sample weight
        
        #P_t : penalty term
        #amb_t : disagreement degree of the classification within the ensemble at current iteration t
                
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')

           


            tree.fit(X_train, y_train,
                   sample_weight=W)

            P = tree.predict(X_train) 
            P_int = P.astype(int) #for indexing class 1 and 0
            
            Prediction = np.ones(N)  #to index negative values
            negative_index = (Prediction != P_int) #indexes of prediction 0 (negative)
            Prediction[negative_index] = -1 #Predicted as negative (0) instances become value -1 positive to 1
            
            y_train_value = np.ones(N) #to index negative values                        
            y_int = y_train.astype(int)
            negative_index_y = (y_train_value != y_int) #index of prediction 0 (negative)
            y_train_value[negative_index_y] = -1 #negative y_train instances become value -1 positive to 1
     
            err = np.sum(W[P != y_train])
            
            if err > 0.5:
                m = m - 1
            elif err <= 0:
                err = 0.0000001
            else:     
      
                if len(self.models) == 0:
                    amb_t = np.zeros(N)
                    P_t = np.ones(N)
                else:
                    FX = np.zeros(N)             
                    
                    for alpha, tree_ in zip(self.alphas, self.models):
                        Prediction_ = np.ones(N) 
                        P_ = tree_.predict(X_train)
                        P_int_ = P_.astype(int)
            
                        negative_index_ = (Prediction_ != P_int_)
                
                        Prediction_[negative_index_] = -1 # instance that have been classified as negative(0) becomes -1
            
                        FX += alpha * Prediction_ # alpha1 * 1 + alphas2 *-1 +...
                    FX = np.sign(FX)  #if fx>0, then fx=1, if fx<0, then fx=-1, if fx=0, then fx=0
        
                    FX[FX == -1] = 0  # prediction result 1 for positive, 0 for negative
                    
                    amb_t = np.zeros(N)
                                
                            
                    for trees in self.models:
                        FX[FX != y_train] = 0
                        FX[FX == y_train] = 1
                        
                        tree_prediction = trees.predict(X_train)
                        tree_prediction[tree_prediction != y_train] = 0
                        tree_prediction[tree_prediction == y_train] = 1
                        
                        difference = FX - tree_prediction
                        
                        amb_t += difference
                        
                    amb_t = amb_t / len(self.models)
                    
                    P_t = np.ones(N) - np.abs(amb_t)
        
                                                                                                                     
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        predicted_wrong_index = (Prediction != y_train_value) 
                        predicted_correct_index = (Prediction == y_train_value)
                        
                        denominator = np.sum(W[predicted_wrong_index]*np.power(P_t[predicted_wrong_index], lambda_))
                        numerator = np.sum(W[predicted_correct_index]*np.power(P_t[predicted_correct_index], lambda_))
                     
                        if denominator >= numerator:
                            alpha = 0
                          
                        else:
                            if denominator <= 0.0001:
                                alpha = 5
                               
                            else: 
                                alpha = 0.5 * (np.log(numerator) - np.log(denominator))
                                
                        
                        
                      #  alpha = 0.5 * (np.log(numerator) - np.log(denominator)) #alpha is assigned vote based on error

                    W = W * np.power(P_t, lambda_) * np.exp(-alpha * y_train_value * Prediction)  
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
        FX = np.sign(FX)
        
        FX[FX==-1] = 0 
            
        
        return FX #0 for negative, 1 for positive



    def predict_proba(self, X_test):
        # if self.alphas == 'SAMME'
        #재호 수정 5/17
        #N, _ = X_test.shape
        #proba = np.zeros(N)
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
