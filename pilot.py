import numpy as np
from scipy import interp
import pandas as pd
import csv
#from sklearn.ensemble import AdaBoostClassifier
#from rusboost import RusBoost
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

from AdaBoost_NC import AdaboostNC_Classifier
from CUSboost_NC import CUSBoostNC_Classifier
from AdaBoost import AdaboostClassifier
from CUSBoost import CUSBoostClassifier

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from imblearn.datasets import fetch_datasets



#dataset = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'car_eval_34', 'us_crime', 'yeast_ml8', 'scene' ,'wine_quality']
dataset = ['ecoli']
#dataset = 'pima.txt'

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

top_auc = 0


#number_of_clusters = 23
#percentage_to_choose_from_each_cluster = 0.5
number_of_clusters_list = [4, 8, 16, 32, 64]
#percentage_to_choose_from_each_cluster_list = [0.2, 0.4, 0.6, 0.8]
lambda_ = [0.25, 0.25*3, 0.25*9, 0.25*27]

estimator = [32, 64, 128]
tree_depth = [1, 2, 4, 8, 16]
Classifiers = [(AdaboostClassifier, "AdaboostClassifier"),(AdaboostNC_Classifier, "AdaboostNC_Classifier"), (CUSBoostClassifier,"CUSBoostClassifier")
, (CUSBoostNC_Classifier, "CUSBoostNC_Classifier")]

random_state_list = [0, 5, 10, 15]
best_adaboost_score = 0
best_adaboost_nc_score = 0
best_cusboost_score = 0
best_cusboost_nc_score = 0

for data in dataset:
    print("dataset : ", data)
    fetch_data = fetch_datasets()[data]
    
    X = fetch_data.data
    y = fetch_data.target
    
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)
    
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    
    #for random_states in random_state_list:
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.3, random_state=0, stratify=y)    
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.25, random_state=0, stratify=y_train_val)
    
    value, counts = np.unique(y, return_counts=True)
    
    if counts[0]>= counts[1]:
        fraction = int((counts[1]/counts[0])*100)
    else: 
        fraction = int((counts[0]/counts[1])*100)
        
    percentage_to_choose_from_each_cluster_list = [fraction/100 , 0.25, 0.5, 0.75]

    
    
    for depth in tree_depth:
        for estimators in estimator:
            '''
            current_param_Adaboost_auc = []
            current_param_AdaboostNC_lambda2_auc = []
            current_param_AdaboostNC_lambda5_auc = []
            current_param_AdaboostNC_lambda8_auc = []
            current_param_AdaboostNC_lambda11_auc = []
            current_param_CUSboost_auc = []
            current_param_CUSboostNC_lambda2_auc = []
            current_param_CUSboostNC_lambda5_auc = []
            current_param_CUSboostNC_lambda8_auc = []
            current_param_CUSboostNC_lambda11_auc = []
            '''
            
            #for i in range(5): # to get mean value of 5tests of 5cv
    
            #for train_index, test_index in skf.split(X, y):
                #X_train_val = X[train_index]
                #X_test = X[test_index]
                
                #y_train_val = y[train_index]
                #y_test = y[test_index]                   
      
            for clf, clf_name in Classifiers:
                if clf_name == "AdaboostClassifier":
                    classifier = clf(depth=depth, n_estimators=estimators)
                    classifier.fit(X_train, y_train) #Adaboost classifier
                    predictions = classifier.predict_proba(X_val)           
                    adaboost_auc = roc_auc_score(y_val, predictions[:, 1])
                    
                    if adaboost_auc > best_adaboost_score:
                        best_adaboost_score = adaboost_auc
                        best_adaboost_hyperparameter = {'depth':depth, 'n_estimators':estimators}
                    
                elif clf_name == "AdaboostNC_Classifier":
                    for lambdas in lambda_:
                        classifier = clf(depth=depth, n_estimators=estimators)
                        classifier.fit(X_train, y_train, lambdas)
                        predictions = classifier.predict_proba(X_val)
                        adaboost_nc_auc = roc_auc_score(y_val, predictions[:, 1])
                        
                        if adaboost_nc_auc > best_adaboost_nc_score:
                            best_adaboost_nc_score = adaboost_nc_auc
                            best_adaboost_nc_hyperparameter = {'depth':depth, 'n_estimators':estimators, 'lambda_':lambdas}
                        '''
                        if lambdas == lambda_[0]:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_AdaboostNC_lambda2_auc.append(auc)
                        elif lambdas == lambda_[1]:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_AdaboostNC_lambda5_auc.append(auc)
                        elif lambdas == lambda_[2]:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_AdaboostNC_lambda8_auc.append(auc)
                        else:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_AdaboostNC_lambda11_auc.append(auc)
                            '''
                elif clf_name == "CUSBoostClassifier":
                    #for clusters in cluster:
                    for cluster_number in number_of_clusters_list:
                        for percentage in percentage_to_choose_from_each_cluster_list:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, cluster_number, percentage) #CUSboost classifier
                            predictions = classifier.predict_proba(X_val)
                            cusboost_auc = roc_auc_score(y_val, predictions[:, 1])
                            
                            if cusboost_auc > best_cusboost_score:
                                best_cusboost_score = cusboost_auc
                                best_cusboost_hyperparameter = {'depth':depth, 'n_estimators':estimators, "number_of_clusters":cluster_number, "percentage_to_choose_from_each_cluster":percentage}
                    
                else:
                     for lambdas in lambda_:
                        for cluster_number in number_of_clusters_list:
                            
                            
                            for percentage in percentage_to_choose_from_each_cluster_list: 
                                classifier = clf(depth=depth, n_estimators=estimators)
                                classifier.fit(X_train, y_train, cluster_number, percentage, lambdas)
                                predictions = classifier.predict_proba(X_val)
                                cusboost_nc_auc = roc_auc_score(y_val, predictions[:, 1])
                                
                                if cusboost_nc_auc > best_cusboost_nc_score:
                                    best_cusboost_nc_score = cusboost_nc_auc
                                    best_cusboost_nc_hyperparameter = {'depth':depth, 'n_estimators':estimators,"number_of_clusters":cluster_number, "percentage_to_choose_from_each_cluster":percentage ,'lambda_':lambdas}
                            
                            '''
                    for lambdas in lambda_:
                        if lambdas == lambda_[0]:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, number_of_clusters ,percentage_to_choose_from_each_cluster ,lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_CUSboostNC_lambda2_auc.append(auc)
                        elif lambdas == lambda_[1]:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, number_of_clusters ,percentage_to_choose_from_each_cluster ,lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_CUSboostNC_lambda5_auc.append(auc)
                        elif lambdas == lambda_[2]:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, number_of_clusters ,percentage_to_choose_from_each_cluster ,lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_CUSboostNC_lambda8_auc.append(auc)
                        else:
                            classifier = clf(depth=depth, n_estimators=estimators)
                            classifier.fit(X_train, y_train, number_of_clusters ,percentage_to_choose_from_each_cluster ,lambdas )
                            predictions = classifier.predict_proba(X_test)
                            auc = roc_auc_score(y_test, predictions[:, 1])
                            current_param_CUSboostNC_lambda11_auc.append(auc)
                                                     '''
                                             
    adaboost_classifier = AdaboostClassifier(depth = best_adaboost_hyperparameter["depth"], n_estimators= best_adaboost_hyperparameter["n_estimators"]) 
    adaboost_nc_classifier = AdaboostNC_Classifier(depth = best_adaboost_nc_hyperparameter["depth"], n_estimators= best_adaboost_nc_hyperparameter["n_estimators"])
    cusboost_classifier = CUSBoostClassifier(depth = best_cusboost_hyperparameter["depth"], n_estimators = best_cusboost_hyperparameter["n_estimators"])
    cusboost_nc_classifier = CUSBoostNC_Classifier(depth = best_cusboost_nc_hyperparameter["depth"], n_estimators= best_cusboost_nc_hyperparameter["n_estimators"])
    
    adaboost_classifier.fit(X_train_val, y_train_val)
    adaboost_nc_classifier.fit(X_train_val, y_train_val, best_adaboost_nc_hyperparameter["lambda_"])
    cusboost_classifier.fit(X_train_val, y_train_val, best_cusboost_hyperparameter["number_of_clusters"], best_cusboost_hyperparameter["percentage_to_choose_from_each_cluster"])
    cusboost_nc_classifier.fit(X_train_val, y_train_val, best_cusboost_nc_hyperparameter["number_of_clusters"], best_cusboost_nc_hyperparameter["percentage_to_choose_from_each_cluster"], best_cusboost_nc_hyperparameter["lambda_"])
    
    adaboost_prediction = adaboost_classifier.predict_proba(X_test)
    final_adaboost_auc = roc_auc_score(y_test, adaboost_prediction[:,1])
    
    adaboost_nc_prediction = adaboost_nc_classifier.predict_proba(X_test)
    final_adaboost_nc_auc = roc_auc_score(y_test, adaboost_nc_prediction[:,1])
    
    cusboost_prediction = cusboost_classifier.predict_proba(X_test)
    final_cusboost_auc = roc_auc_score(y_test, cusboost_prediction[:,1])
    
    cusboost_nc_prediction = cusboost_nc_classifier.predict_proba(X_test)
    final_cusboost_nc_auc = roc_auc_score(y_test, cusboost_nc_prediction[:,1])
    

            
            #top auc of depth =? estimator=? 를 저장하는 코드
    f = open('result4_pilottest_{0}.csv'.format(data),'w', newline='')
    with f:
        writer = csv.writer(f)
        writer.writerow([data,best_adaboost_hyperparameter,"adaboost_auc", final_adaboost_auc])
        writer.writerow([data,best_adaboost_nc_hyperparameter,"adaboost_nc_auc", final_adaboost_nc_auc])
        writer.writerow([data,best_cusboost_hyperparameter,"cusboost_auc", final_cusboost_auc])
        writer.writerow([data,best_cusboost_nc_hyperparameter,"cusboost_nc_auc", final_cusboost_nc_auc])
        

        
    print('result_3_pilottest_{0} have been completed'.format(data)) 
                    
         
            
            
            #if top_auc < current_mean_auc:
            #    top_auc = current_mean_auc
        
             #   best_depth = depth
              #  best_estimators = estimators
                        
               # best_auc = top_auc
                #best_aupr = current_mean_aupr
        
                #best_tpr = np.mean(tprs, axis=0)
                #best_fpr = mean_fpr
        
       #         best_precision, best_recall, _ = precision_recall_curve(y_test, predictions[:, 1])
        #        best_fpr, best_tpr, thresholds = roc_curve(y_test, predictions[:, 1])
        
       
            
#print(prediction_)


#print('ploting', dataset)
#    plt.clf()
#plt.plot(best_recall, best_precision, lw=2, color='Blue',
#         label='Precision-Recall Curve')
#plt.plot(best_fpr, best_tpr, lw=2, color='red',
#         label='ROC curve')

#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.legend(loc="upper right")
#plt.show()

# plt.plot(fpr_c[1], tpr_c[1], lw=2, color='red',label='Roc curve: Clustered sampling')


