# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:08:11 2022

@author: Ommarselim
"""


# Task3:
#         Diabetes data set at Kaggle
#         https://www.kaggle.com/uciml/pima-indians-diabetes-database 
# 1.	Use the diabetes dataset to apply the classification algorithms: Decision  Tree, Random Forest, KNN, and Logistic Regression. 
# 2.	Evaluate each algorithm and compare between all algorithms.
# 3.	Finally discuss your conclusion about these algorithms with the dataset.



#imports 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy.sparse.sputils import matrix




dataset = pd.read_csv("diabetes.csv")
print(dataset.head())

diabetes_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction","Age"]
X = dataset[diabetes_features]
diabetes_output = ["Outcome"]
y = dataset[diabetes_output]
print(X.head())

print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.2)


# Clean data from missing values >>
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns


# fit the and find error



diabetes_model = DecisionTreeRegressor()
diabetes_model.fit(imputed_X_train, y_train)
X_validation = diabetes_model.predict(imputed_X_test)
decision_tree_error = mean_absolute_error( y_test ,X_validation)
print(decision_tree_error)


#Random forest
forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_forest = forest_model.predict(imputed_X_test)
random_forest_error = mean_absolute_error(y_test, X_validation_forest)
print(random_forest_error)


#Logistic 
logistic_model = LogisticRegression(max_iter= 180)
logistic_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_logistic = logistic_model.predict(imputed_X_test)
logistic_error = mean_absolute_error(y_test, X_validation_logistic)
print(logistic_error)


#KNN 


#feature scaling 
X_scale = StandardScaler()
imputed_X_train = X_scale.fit_transform(imputed_X_train)
imputed_X_test = X_scale.transform(imputed_X_test)




KNN_model = KNeighborsClassifier(n_neighbors= 27, p=2, metric='euclidean')
KNN_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_KNN = KNN_model.predict(imputed_X_test)

O = confusion_matrix(y_test, X_validation_KNN)


#printing confusion matrix
print(O)


print(f"the value of f1 score by KNN is ==> {f1_score(y_test, X_validation_KNN)}")


print(f"the accuracy by KNN is ==> {accuracy_score(y_test, X_validation_KNN)} ")


print(f"the value of f1 score by decision tree is  ==> {f1_score(y_test, X_validation)}")


print(f"the accuracy by decision tree is ==> {accuracy_score(y_test, X_validation)} ")


print(f"the value of f1 score by logistic is ==> {f1_score(y_test, X_validation_logistic)}")


print(f"the accuracy by logistic is ==> {accuracy_score(y_test, X_validation_logistic)} ")




