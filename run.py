import numpy as np
from scipy import interp
import pandas as pd
#from sklearn.ensemble import AdaBoostClassifier
#from rusboost import RusBoost
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import csv
from AdaBoost_NC import AdaboostNC_Classifier
from CUSboost_NC import CUSBoostNC_Classifier
from AdaBoost import AdaboostClassifier
from CUSBoost import CUSBoostClassifier
from rusboost import RusBoostClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from imblearn.datasets import fetch_datasets
from preprocessing_KEEL import preprocessing_KEEL
from preprocessing_UCI import preprocessing_UCI
# datasets = ['gpcr_dataset_1282.txt' ]
#dataset = 'pima.txt'
#dataset = ["ecoli"]
    
'''
print("dataset : ", dataset)
df = pd.read_csv(dataset, header=None)
df['label'] = df[df.shape[1] - 1]
#
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label']) #positive to 1, negative to 0
#
X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

normalization_object = Normalizer()
X = normalization_object.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True)

top_auc = 0
mean_fpr = np.linspace(0, 1, 100)
number_of_clusters = 23
percentage_to_choose_from_each_cluster = 0.5


current_param_f1 = []
current_param_auc = []
#current_param_aupr = []
#tprs = []
'''
dataset = ["yeast_ml8"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
number_of_clusters = 23

a = {'depth':4, "n_estimators":64
     }
for data in dataset:
    print("dataset : ", data)
    '''
    fetch_data = fetch_datasets()[data]
    
    X = fetch_data.data
    y = fetch_data.target
    
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)
    
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    '''
    fetch_data = fetch_datasets()[data]
    
    
    X = fetch_data.data
    y = fetch_data.target
    
    Standard_object = StandardScaler()
    X = Standard_object.fit_transform(X)
    
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    value, counts = np.unique(y, return_counts=True)
    
    if counts[0]>= counts[1]:
        fraction = int((counts[1]/counts[0])*100)
    else: 
        fraction = int((counts[0]/counts[1])*100)
    
    current_param_auc = []
    current_param_f1 = []
    current_param_accuracy = []
    

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        
        y_train = y[train_index]
        y_test = y[test_index]
    
       
    
    
    
        
        
        
        #classifier = CUSBoostClassifier(**a) 
        #classifier = AdaboostClassifier(**a)
        #classifier = RusBoost(depth=depth, n_estimators=estimators)
        #classifier = AdaboostNC_Classifier(**a)
        #classifier = CUSBoostNC_Classifier(**a)
        #classifier = RusBoost(**a)
        classifier = RUSBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=64)
    
        #classifier.fit(X_train, y_train, number_of_clusters, 0.5) #CUSBoost classifier        
        #classifier.fit(X_train, y_train) #Adaboost classifier
        #classifier.fit(X_train, y_train, 0.5) #AdaboostNC classifier
        #classifier.fit(X_train, y_train, 6, 0.5)
        #classifier.fit(X_train, y_train, 6, fraction/100, 8)
        classifier.fit(X_train, y_train)
        
        
        
        predictions = classifier.predict_proba(X_test)
        prediction_ = classifier.predict(X_test)
    
        auc = roc_auc_score(y_test, predictions[:, 1])
        f1 = f1_score(y_test, prediction_)
        accuracy = accuracy_score(y_test, prediction_)
    
        #aupr = average_precision_score(y_test, predictions[:, 1])
    
        current_param_auc.append(auc)
        current_param_f1.append(f1)
        current_param_accuracy.append(accuracy)
    
        #current_param_aupr.append(aupr)
    
        #fpr, tpr, thresholds = roc_curve(y_test, predictions[:, 1])
        #tprs.append(interp(mean_fpr, fpr, tpr))
        #tprs[-1][0] = 0.0
    
    current_mean_auc = np.mean(np.array(current_param_auc))
    #current_mean_aupr = np.mean(np.array(current_param_aupr))
    current_mean_f1 = np.mean(np.array(current_param_f1))
    current_mean_accuracy = np.mean(np.array(current_param_accuracy))
    '''
    if top_auc < current_mean_auc:
        top_auc = current_mean_auc
        
        best_f1 = current_mean_f1
    '''
    
    
    print('Mean_ROC: ', current_mean_auc, 'f1:', current_mean_f1, "accuracy", current_mean_accuracy)
    
    f = open('result_small_tree_{}.csv'.format(data),'w', newline='')
    with f:
        writer = csv.writer(f)
        writer.writerow([dataset,"adaboost_auc", current_mean_auc, "f1", current_mean_f1])
    
    print('result_{0}_pilot_adaboost'.format(data)) 
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


