import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import rankdata
import networkx as nx
import pandas as pd
from collections import defaultdict,Counter
from datetime import datetime, date
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from features_utils import *
 

def prepare_split_datasets(full_train_data, user_parameter, logs_file_name):
    """
    Split the whole dataset (unconnected pairs) into two classes, positive and negative samples 

    Parameters:
    - full_train_data: full dataset for training or testing
    - user_parameter: some user parameters whcih are num_class, IR_num, split_type, out_norm
    - logs_file_name: the folder to store the figure, which is usually defined from t0_c2_curve created from make_folders()

    return:
    data_subsets: negative and postive samples
    """

    num_class, IR_num, split_type, out_norm = user_parameter

    time_start=time.time()
    data_subsets = []
    if split_type==0: ## IR_num=[100]; namely citation >=IR and citation<IR; binary case 1
        data_subsets.append(full_train_data[full_train_data[:, 2] < IR_num[0]]) # get all the negative samples, which are the citations < IR_num[0]
        data_subsets.append(full_train_data[full_train_data[:, 2] >= IR_num[0]]) # get all the positive samples, which are the citations >= IR_num[0]
    else:  # conditional case
        for i in range(len(IR_num)-1): # e.g., IR_num=[[0,5], 100]; namely citation >=100 and citation in [0,5] 
            data_subsets.append(full_train_data[(full_train_data[:, 2] >= IR_num[i][0]) & (full_train_data[:, 2] <= IR_num[i][1])])
        data_subsets.append(full_train_data[full_train_data[:, 2] >= IR_num[-1]])
        
    if split_type==0: 
        if len(data_subsets[0])<=3*10**7: # usually not meet, due to the size of data_subsets[0] is more than 600million
            num_row_chose=len(data_subsets[0])
        else:
            num_row_chose=min(len(data_subsets[0]),len(data_subsets[1])) # take the same size as the positive cases
    else:  # conditional case
        num_row_chose=len(data_subsets[0])
        
    indices = np.random.choice(data_subsets[0].shape[0], size=num_row_chose, replace=False)  
    data_subsets[0] = data_subsets[0][indices] # randomly chose num_row_chose negative samples
    
    print(f"dataset len: {len(data_subsets[0])}, {len(data_subsets[1])}; num_row_chose: {num_row_chose}; {time.time()-time_start}s")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\ndataset len: {len(data_subsets[0])}, {len(data_subsets[1])}; num_row_chose: {num_row_chose}; {time.time()-time_start}s")
          
        
    return data_subsets


def shuffle_split_datasets(data_subsets, train_valid_test_size):
    """
    Split the dataset into training and testing sets

    Parameters:
    - data_subsets: the prepared negative and positive samples (unconnected pairs)
    - train_valid_test_size: split ratio, here is train_valid_test_size=[0.85, 0.15, 0.0]; 85% for training, 15% for testing; evaluation part is using future data
 
    return:
    dataset_train: training dataset
    dataset_test: testing dataset
    """
    dataset_train = []
    dataset_test = []
    for subset in data_subsets:
        np.random.shuffle(subset)
        idx_train = int(len(subset) * train_valid_test_size[0])
        train_set = subset[:idx_train]
        test_set = subset[idx_train:]
        dataset_train.append(train_set)
        dataset_test.append(test_set)
    
    return dataset_train, dataset_test


def get_pair_solution_datasets(data_subsets, hyper_parameter, user_parameter, logs_file_name):
    """
    prepare the unconnected pairs in year y and their corresponding future citations solutions

    Parameters:
    - data_subsets: the prepared negative and positive samples (unconnected pairs)
    - hyper_parameter:  batch_size, lr_enc, rnd_seed
    - user_parameter: not used here
    - logs_file_name: txt file for logging the running status 

    return:
    train_edge_pair: dataset (unconnected pairs)
    train_edge_solution: the corresponding citations 
    """    
    num_class, IR_num, split_type, out_norm = user_parameter # not used here
    batch_size, lr_enc, rnd_seed=hyper_parameter
    start_time=time.time()   

    for idx, subset in enumerate(data_subsets):

        min_num_row=min(batch_size, len(subset))
        num_new_samples = len(subset) - min_num_row ## data_subsets[0]>=batch_size
        if num_new_samples<0:  # data_subsets[0]<batch_size, one can set the batch_size smaller such that this part will not meet
            upsamples=np.abs(num_new_samples)
            new_samples_idx = np.random.choice(subset.shape[0], size=upsamples, replace=True)
            data_subsets[idx]= np.concatenate([subset, subset[new_samples_idx]], axis=0)
            
    train_data_for_checking = np.concatenate(data_subsets, axis=0)
    train_edge_pair=train_data_for_checking[:,:2]
    train_edge_solution=train_data_for_checking[:,2]
    
    print(f"\nDone, data_for_checking: {len(train_data_for_checking)}; elapsed_time: {time.time() - start_time}") 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, data_for_checking: {len(train_data_for_checking)}; elapsed_time: {time.time() - start_time}")
        
    return train_edge_pair, train_edge_solution


    
def prepare_train_data(data_pair, data_solution, all_features, user_parameter, logs_file_name):
    """
    prepare the features for unconnected pairs and separate into two classes whcih are used in train_model()

    Parameters:
    - data_pair: unconnected pairs
    - data_solution:  the citation solution
    - all_features: all the types of features, e.g., all_features=[node_feature, node_cfeature, pair_feature_train, pair_cfeature_train]
    - user_parameter: num_class, IR_num, split_type, out_norm
    - logs_file_name: txt file for logging the running status 

    return:
    data_feature: the 141 features for each unconnected pair, 2-d numpy array, row is for each unconnected pair, 141 columns for the 141 features
    data_input:  separate features ; data_input[0] for negative samples; data_input[1] for postive samples
    """ 

    num_class, IR_num, split_type, out_norm = user_parameter
    data_feature=get_all_feature(all_features, data_pair, logs_file_name) # make all 141 features for each unconnected pair
    
    data_input = []
    start_time=time.time()
    if split_type==0:
        data_input.append(data_feature[data_solution < IR_num[0]]) # features for negative samples
        data_input.append(data_feature[data_solution >= IR_num[0]]) # features for positive samples         
    else: # conditional case
        for i in range(len(IR_num) - 1):
            data_input.append(data_feature[(data_solution >= IR_num[i][0]) & (data_solution <=IR_num[i][1])])
        data_input.append(data_feature[data_solution >= IR_num[-1]])
        
    print(f"\n   finish split_data_features: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n   finish split_data_features: {time.time()-start_time}")    
    
    return data_feature, data_input
 

    
######### classify_solution ###############
def classify_solution(data_solution, user_parameter):
    """
    classfiy the citation solution into 0 and 1 classes

    Parameters:
    - data_solution:  the citation solution
    - user_parameter: num_class, IR_num, split_type, out_norm

    return:
    solution_arr: 0-1 numpy array, 0 for negative samples, 1 for postive samples
    """     
    num_class, IR_num, split_type, out_norm = user_parameter
    solution_arr = np.zeros_like(data_solution)
    
    if split_type==0: ## only binary
        solution_arr[data_solution < IR_num[0]] = 0
        solution_arr[data_solution >= IR_num[0]] = 1
        
    else: # conditional case
        for i in range(len(IR_num) - 1): # in this work, IR_num=[[0,5], 100]
            solution_arr[(data_solution >= IR_num[i][0]) & (data_solution <=IR_num[i][1])] = i # in this case, 0
        solution_arr[data_solution >= IR_num[-1]] = len(IR_num) - 1 # in this case, 1

    return solution_arr





