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
from sklearn.model_selection import StratifiedKFold

from imblearn.datasets import fetch_datasets




dataset = ['ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'car_eval_34', 'us_crime', 'yeast_ml8', 'scene' ,'wine_quality']

#dataset = 'pima.txt'

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

top_auc = 0
mean_fpr = np.linspace(0, 1, 100)

number_of_clusters = 23
percentage_to_choose_from_each_cluster = 0.5
lambda_ = [0.25, 0.75, 0.75*3, 0.75*9, 0.75*15]

Classifiers = [(AdaboostClassifier, "AdaboostClassifier"),(AdaboostNC_Classifier, "AdaboostNC_Classifier"), (CUSBoostClassifier,"CUSBoostClassifier")
, (CUSBoostNC_Classifier, "CUSBoostNC_Classifier")]

for data in dataset:
    print("dataset : ", data)
    fetch_data = fetch_datasets()[data]
    
    X = fetch_data.data
    y = fetch_data.target
    
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)
    
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    
    for depth in range(2, 33, 10):
        for estimators in range(20, 85, 30):
    
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
         
            
            #for i in range(5): # to get mean value of 5tests of 5cv
    
            for train_index, test_index in skf.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]
                
                y_train = y[train_index]
                y_test = y[test_index]
                
                for clf, clf_name in Classifiers:
                    if clf_name == "AdaboostClassifier":
                        classifier = clf(depth=depth, n_estimators=estimators)
                        classifier.fit(X_train, y_train) #Adaboost classifier
                        predictions = classifier.predict_proba(X_test)
                        auc = roc_auc_score(y_test, predictions[:, 1])
                        current_param_Adaboost_auc.append(auc)
                    elif clf_name == "AdaboostNC_Classifier":
                        for lambdas in lambda_:
                            if lambdas == 2:
                                classifier = clf(depth=depth, n_estimators=estimators)
                                classifier.fit(X_train, y_train, lambdas )
                                predictions = classifier.predict_proba(X_test)
                                auc = roc_auc_score(y_test, predictions[:, 1])
                                current_param_AdaboostNC_lambda2_auc.append(auc)
                            elif lambdas == 5:
                                classifier = clf(depth=depth, n_estimators=estimators)
                                classifier.fit(X_train, y_train, lambdas )
                                predictions = classifier.predict_proba(X_test)
                                auc = roc_auc_score(y_test, predictions[:, 1])
                                current_param_AdaboostNC_lambda5_auc.append(auc)
                            elif lambdas == 8:
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
                    elif clf_name == "CUSBoostClassifier":
                        classifier = clf(depth=depth, n_estimators=estimators)
                        classifier.fit(X_train, y_train, number_of_clusters, percentage_to_choose_from_each_cluster) #CUSboost classifier
                        predictions = classifier.predict_proba(X_test)
                        auc = roc_auc_score(y_test, predictions[:, 1])
                        current_param_CUSboost_auc.append(auc)
                    else:
                        for lambdas in lambda_:
                            if lambdas == 2:
                                classifier = clf(depth=depth, n_estimators=estimators)
                                classifier.fit(X_train, y_train, number_of_clusters ,percentage_to_choose_from_each_cluster ,lambdas )
                                predictions = classifier.predict_proba(X_test)
                                auc = roc_auc_score(y_test, predictions[:, 1])
                                current_param_CUSboostNC_lambda2_auc.append(auc)
                            elif lambdas == 5:
                                classifier = clf(depth=depth, n_estimators=estimators)
                                classifier.fit(X_train, y_train, number_of_clusters ,percentage_to_choose_from_each_cluster ,lambdas )
                                predictions = classifier.predict_proba(X_test)
                                auc = roc_auc_score(y_test, predictions[:, 1])
                                current_param_CUSboostNC_lambda5_auc.append(auc)
                            elif lambdas == 8:
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
                                                     
                                                                                                                       
            #current_mean_auc = np.mean(np.array(current_param_auc))
            current_mean_Adaboost_auc = np.mean(np.array(current_param_Adaboost_auc))
            current_mean_AdaboostNC_lambda2_auc = np.mean(np.array(current_param_AdaboostNC_lambda2_auc))
            current_mean_AdaboostNC_lambda5_auc = np.mean(np.array(current_param_AdaboostNC_lambda5_auc))
            current_mean_AdaboostNC_lambda8_auc = np.mean(np.array(current_param_AdaboostNC_lambda8_auc))
            current_mean_AdaboostNC_lambda11_auc = np.mean(np.array(current_param_AdaboostNC_lambda11_auc))
            current_mean_CUSboost_auc = np.mean(np.array(current_param_CUSboost_auc))
            current_mean_CUSboostNC_lambda2_auc = np.mean(np.array(current_param_CUSboostNC_lambda2_auc))
            current_mean_CUSboostNC_lambda5_auc = np.mean(np.array(current_param_CUSboostNC_lambda5_auc))
            current_mean_CUSboostNC_lambda8_auc = np.mean(np.array(current_param_CUSboostNC_lambda8_auc))
            current_mean_CUSboostNC_lambda11_auc = np.mean(np.array(current_param_CUSboostNC_lambda11_auc))
        
            
            #top auc of depth =? estimator=? 를 저장하는 코드
            f = open('result2_checkIfAllSameOccurs_{0}_depth{1}_n_estimator{2}.csv'.format(data,depth,estimators),'w', newline='')
            with f:
                writer = csv.writer(f)
                writer.writerow([data,depth,estimators,"current_mean_Adaboost_auc", current_mean_Adaboost_auc])
                writer.writerow([data,depth,estimators,"current_mean_AdaboostNC_lambda2_auc", current_mean_AdaboostNC_lambda2_auc])
                writer.writerow([data,depth,estimators,"current_mean_AdaboostNC_lambda5_auc", current_mean_AdaboostNC_lambda5_auc])
                writer.writerow([data,depth,estimators,"current_mean_AdaboostNC_lambda8_auc", current_mean_AdaboostNC_lambda8_auc])
                writer.writerow([data,depth,estimators,"current_mean_AdaboostNC_lambda11_auc", current_mean_AdaboostNC_lambda11_auc])
                writer.writerow([data,depth,estimators,"current_mean_CUSboost_auc", current_mean_CUSboost_auc])
                writer.writerow([data,depth,estimators,"current_mean_CUSboostNC_lambda2_auc", current_mean_CUSboostNC_lambda2_auc])
                writer.writerow([data,depth,estimators,"current_mean_CUSboostNC_lambda5_auc", current_mean_CUSboostNC_lambda5_auc])
                writer.writerow([data,depth,estimators,"current_mean_CUSboostNC_lambda8_auc", current_mean_CUSboostNC_lambda8_auc])
                writer.writerow([data,depth,estimators,"current_mean_CUSboostNC_lambda11_auc", current_mean_CUSboostNC_lambda11_auc])
                
            print('result_{0}_depth{1}_n_estimator{2} have been completed'.format(data,depth,estimators)) 
                    
         
            
            
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




