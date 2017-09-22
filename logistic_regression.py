#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:59:43 2017

@author: akashmantry
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange

"""
   Benign: 458 (65.5%)
   Malignant: 241 (34.5%)
   16 missing data points
   0 beningn
   1 malignant
"""
dataset = pd.read_csv('breast_cancer.csv', sep='\t')
dataset.replace('?', np.nan, inplace=True)
dataset.drop(['ID'], 1, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.apply(pd.to_numeric, errors='ignore')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

#==============================================================================
#from sklearn import cross_validation, linear_model
# 
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# 
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#
#clf = linear_model.LogisticRegression()
#clf.fit(X_train, y_train)
#accuracy = clf.score(X_test, y_test)
#print(accuracy)
#==============================================================================

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    cross_validation_dataset = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset_copy)/n_folds)
    
    for i in range(0, n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        cross_validation_dataset.append(fold)
    return cross_validation_dataset

def sigmoid(X):
    return 1/(1 + np.exp(- X))
    
def pre_processing(matrix):
    range_ = 10
    b = np.apply_along_axis(lambda x: (x-np.mean(x))/range_, 0, matrix)
    return b

def cost_function(X, y, theta):
    h_theta = sigmoid(np.dot(X, theta))
    log_l = (-y)*np.log(h_theta) + (1 - y)*np.log(1 - h_theta)
    return log_l.mean()

def calculate_gradient(X, y, theta, index, X_count):
    dummy_theta = sigmoid(np.dot(X, theta))
    sum_ = 0.0
    for i in range(dummy_theta.shape[0]):
        sum_ = sum_ + (dummy_theta[i] - y[i]) * X[i][index]
    return sum_


def gradient_descent(training_set, alpha, max_iterations, plot_graph):
    iter_count = 0

    training_set = np.asarray(training_set)
    X = training_set.T[0:9].T
    y = training_set.T[9].T
    X_count = X.shape[1]

    theta = np.zeros(X_count)
    x_vals = []
    y_vals = []
    while(iter_count < max_iterations):
        iter_count += 1
        for i in range(X_count):
            prediction = calculate_gradient(X, y, theta, i, X_count)
            prev_theta = theta[i]
            theta[i] = prev_theta - alpha * prediction
            
            mean = cost_function(X, y, theta)
            x_vals.append(iter_count)
            y_vals.append(mean)
    
    if plot_graph:
        plt.suptitle("Gradient Descent plot")
        plt.gca().invert_yaxis()
        plt.plot(x_vals,y_vals)
        plt.xlabel("Iteration ")
        plt.ylabel("Cost function J(theta)")
        #plt.show()
        fileName = "gradient_descent_plot.png";
        plt.savefig(fileName, bbox_inches='tight')
    
    return theta
            
def compute_efficiency(test_set, theta):
    test_set = np.asarray(test_set)
    X = test_set.T[0:9].T
    y = test_set.T[9].T
    X_count = X.shape[0]
    correct = 0
    
    for i in range(X_count):
        prediction = 0
        value  = np.dot(theta, X[i])
        if value >= 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction == y[i]:
            correct+=1
    return correct*100/X.shape[0]
    
    
def evaluate_algorithm(dataset, n_folds, alpha, max_iterations, plot_graph):
    folds = cross_validation_split(dataset, n_folds)
    results = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
        
        theta = gradient_descent(train_set, alpha, max_iterations, plot_graph)
        results.append(compute_efficiency(test_set, theta))
    return np.asarray(results)


X = pre_processing(X)
reshaped_y = y.reshape(y.shape[0], -1)
processed_dataset = np.concatenate((X, reshaped_y), axis=1)
results = evaluate_algorithm(processed_dataset.tolist(), n_folds=10,
                             alpha=0.01, max_iterations=500, plot_graph=True)
print("Mean : ",np.mean(results)) 
