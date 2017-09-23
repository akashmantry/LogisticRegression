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


dataset = pd.read_csv('spambase.csv')
dataset.replace('?', np.nan, inplace=True)
dataset.drop(['ID'], 1, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.apply(pd.to_numeric, errors='ignore')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

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
    matrix_copy = matrix     
    b = np.apply_along_axis(lambda x: (x-np.mean(x))/float(np.std(x)),0,matrix_copy)
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
    X = training_set.T[0:57].T
    y = training_set.T[57].T
    X_count = X.shape[1]

    theta = np.zeros(X_count)
    x_vals = []
    y_vals = []
    regularization_parameter = 1
    while(iter_count < max_iterations):
        iter_count += 1
        for i in range(X_count):
            prediction = calculate_gradient(X, y, theta, i, X_count)
            prev_theta = theta[i]
            if i != 0:
                prediction += (regularization_parameter/X_count)*prev_theta
            theta[i] = prev_theta - alpha * prediction
            
            if plot_graph:
                mean = cost_function(X, y, theta)
                x_vals.append(iter_count)
                y_vals.append(mean)
    
    if plot_graph:
        plt.suptitle("Gradient Descent plot")
        plt.plot(x_vals,y_vals)
        plt.xlabel("Iteration ")
        plt.ylabel("Cost function J(theta)")
        plt.show()
        #fileName = "gradient_descent_plot.png";
        #plt.savefig(fileName, bbox_inches='tight')
    
    return theta
            
def compute_efficiency(test_set, theta):
    test_set = np.asarray(test_set)
    X = test_set.T[0:57].T
    y = test_set.T[57].T
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
                             alpha=.00004, max_iterations=100, plot_graph=True)
print("Mean : ",np.mean(results))
