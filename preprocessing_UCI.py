# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:26:08 2020

@author: Jaeho
"""
#dataset = "glass"

import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler


def preprocessing_UCI(dataset):
    with open('./KEEL_dataset/{}.data'.format(dataset), 'r+') as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if '@' not in line:
                f.write(line)
                f.truncate()
                
        data = np.genfromtxt(new_f, delimiter=',',  dtype='str')      
       
        X = data[:,:-1]
        
        _, X_column_shape = X.shape
        
        y = data[:,-1]
        
        standardize = StandardScaler()
        
        labelencoder = LabelEncoder()
        
        y= labelencoder.fit_transform(y)
        
        try:
            
            X = X.astype(np.float)
            
            X = standardize.fit_transform(X)
            
          
        except:
            
            for X_columns in range(X_column_shape):
                try:
                    X[:,[X_columns]] = X[:,[X_columns]].astype(np.float)
                    
                    #X[:,X_columns] = np.squeeze(standardize.fit(X[:,X_columns].reshape(-1,1)))
                    #print("try", X_columns)
                    
                
                except:
                    X[:,X_columns] = labelencoder.fit_transform(X[:,X_columns])
                    #print("except",X_columns)
                    
                    
        
                    
                    
                    
            
        
            
            
        
        
    return X, y

#dataset = ["pima","led7digit-0-2-4-5-6-7-8-9_vs_1", "poker-9_vs_7", "segment0", "abalone9-18", "yeast5"]  for binary classification




#dataset = ["pima", "dermatology", "dermatology", "led7digit-0-2-4-5-6-7-8-9_vs_1", "abalone9-18", "yeast", "poker-9_vs_7", "kddcup-guess_passwd_vs_satan", 
#           "yeast5", "ecoli", "]abalone19", "pageblock", "shuttle"]       
    
#dataset = ["pima","led7digit-0-2-4-5-6-7-8-9_vs_1", "poker-9_vs_7", "yeast5"(problem),"segment0"(problem), abalone9-18(problem)] 

#x, y = preprocessing_KEEL("led7digit-0-2-4-5-6-7-8-9_vs_1")
#dataset = ["pima","led7digit-0-2-4-5-6-7-8-9_vs_1", "poker-9_vs_7", "segment0", "abalone9-18", "yeast5"]    #binary set, no string features      

 

#for data in dataset:
 #   x, y = preprocessing_KEEL(data)
        #data has categorical features
        #abalone9-18, abalone19 (attribute has f or m)kddcup, 
        #yeast, shuttle, ecoli, pageblocks, has multiclass
        
        #binary datasets = pima, segment0, led7digit, abalone9-18(categorical feature), pokwer -9vs7, kddcup-guess(categorical feature), yeast5, abalone19(categorical)
        
      
        
    
    
    