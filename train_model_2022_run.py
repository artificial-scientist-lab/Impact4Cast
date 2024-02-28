import os
import pickle
import gzip
import copy
import torch
from torch import nn
import torch.nn.functional as F
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
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from general_utils import *
from preprocess_utils import *
from features_utils import *
from train_model_utils import * 

 

rn_time=random.random()*30
time.sleep(rn_time)
    
if __name__ == '__main__':

    
    split_type=0 # 1 is for conditional case
    out_norm=False # we fix this to False, using the raw scores from the neural network output
    num_class=2 # binary classfication, fixed 
    day_origin = date(1990,1,1) # the baseline time

    vertex_degree_cutoff=1 # fixed, the vertex has at least one edge connecting to it
    min_edges=1 # fixed, minimal number of edges that is considered, not used in the work, can be removed
    years_delta=3 # year gap is 3 years
    year_start=2022-years_delta # train 2019 for 2022
    
    graph_parameter=[year_start,years_delta,vertex_degree_cutoff, min_edges] # parameters for the knowledge graph
    
    # create folders and subfolders
    # it will create a main folder: 2019_train that contains subfolders: t0_c2_log, t0_c2_net, t0_c2_loss, t0_c2_curve, t0_c2_result 
    save_folders, log_folder=make_folders(year_start, split_type, num_class, "train")
    
    log_run=os.path.join(log_folder,f"train_model_{year_start+years_delta}_run_1") # just a log file to check the running status
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\n\nstart: {datetime.now()}\n")    
        
    # load the full dynamic graph
    start_time = time.time()
    data_folder="data_concept_graph" # folder that stores the full knowledge graph
    graph_file=os.path.join(data_folder,"full_dynamic_graph.parquet")
    full_dynamic_graph = pd.read_parquet(graph_file)
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, read full_dynamic_graph: {len(full_dynamic_graph)}; elapsed_time: {time.time() - start_time}")
    
    # load data for preparing different type of features
    feature_folder="data_for_features" # folder that stores data used for preparing features
    start_time=time.time()
    adj_mat_sparse=[]
    node_neighbor_list=[]
    num_neighbor_list=[]
    for yy in [year_start,year_start-1,year_start-2]:
        data_file=os.path.join(feature_folder, f"adjacency_matrix_{yy}.gz") # load the adjacency_matrix file, if not exists, it will automatically generate one
        adj_mat=get_adjacency_matrix(full_dynamic_graph, yy, data_file) # adjacency_matrix for year y, y-1, y-2
        adj_mat_sparse.append(adj_mat)

        curr_node_neighbor=get_node_neighbor(adj_mat)
        node_neighbor_list.append(curr_node_neighbor) # get the neighbors of vertices for years y, y-1, y-2

        curr_num_neighbor = np.array(adj_mat.sum(axis=0)).flatten()  
        num_neighbor_list.append(curr_num_neighbor) # get the number of neighbors of vertices for years y, y-1, y-2
    
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, adjacency_matrix_sparse; elapsed_time: {time.time() - start_time}")
    
    start_time=time.time()
    vertex_features=get_all_node_feature(adj_mat_sparse, year_start, feature_folder) # prepare all the node features for a vertex in years y, y-1, y-2
    
    # load data for preparing different type of citation features
    start_time=time.time()
    vc_feature_list=[]
    for yy in [year_start,year_start-1,year_start-2]:
        data_file=os.path.join(feature_folder, f"concept_node_citation_data_{yy}.parquet") # load the citation information for concepts in year yy
        vc_df=pd.read_parquet(data_file)
        vc_feature=vc_df.values
        vc_feature_list.append(vc_feature)

    vertex_cfeatures=get_all_node_cfeature(vc_feature_list)
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, vertex_cfeatures; elapsed_time: {time.time() - start_time}")
        
    pair_cf_parameter=[vc_feature_list, node_neighbor_list, num_neighbor_list, vertex_features, vertex_cfeatures] # later used for pair features and cfeatures
            
    # load the whole unconnected pairs for training and testing
    train_data_folder = 'data_pair_solution'  
    train_pair_file1=os.path.join(train_data_folder,f"unconnected_{year_start}_pair_solution_connected_{year_start+years_delta}_clean.parquet")
    train_pair_file2=os.path.join(train_data_folder,f"unconnected_{year_start}_pair_solution_unconnected_{year_start+years_delta}.parquet")

    time_start = time.time()
    train_pair_data_yes = pd.read_parquet(train_pair_file1)
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, read unconnected_{year_start}_pair_solution_connected_{year_start+years_delta}: {len(train_pair_data_yes)}; elapsed_time: {time.time() - time_start}")

    time_start = time.time()
    train_pair_data_no = pd.read_parquet(train_pair_file2)
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, read unconnected_{year_start}_pair_solution_unconnected_{year_start+years_delta}: {len(train_pair_data_no)}; elapsed_time: {time.time() - time_start}")

    time_start = time.time()
    full_train_data=np.concatenate((train_pair_data_yes.values, train_pair_data_no.values), axis=0)
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\nDone, combine all: {len(full_train_data)}; elapsed_time: {time.time() - time_start}") 

    full_dynamic_graph=pd.DataFrame()
    train_pair_data_yes=pd.DataFrame()
    train_pair_data_no=pd.DataFrame()
 
    # train neural networks for different IR number from 1 to 200
    IR_start=1  
    IR_end=40 # IR_end=200
    IR_count=IR_start
    while IR_count <= IR_end:
        
        num_impact=random.randint(IR_start, IR_end) # used for parallel computing  
        IR_num=[num_impact]
        IR_Str=format_IR(IR_num, split_type)

        logs_file_name=os.path.join(log_folder,f"train_model_{year_start+years_delta}_"+IR_Str) 
        if not os.path.exists(logs_file_name+"_logs.txt"):
            current_time=datetime.now()
            open(logs_file_name+"_logs.txt", 'a').close()

            batch_size=1000 
            lr_enc=3*10**-5
            rnd_seed=42
            hyper_parameter=[batch_size, lr_enc, rnd_seed]
            graph_parameter=[year_start,years_delta,vertex_degree_cutoff, min_edges]
            user_parameter=[num_class, IR_num, split_type, out_norm]
            
            impact_classfication(full_train_data, [], [], pair_cf_parameter, hyper_parameter, graph_parameter, user_parameter, save_folders, logs_file_name)
            
            IR_count+=1
            rn_time=random.random()*30
            time.sleep(rn_time)
            
        else:
            pass
    
    with open(log_run+"_logs.txt", "a") as myfile:
        myfile.write(f"\nfinish: {datetime.now()}\n\n") 
            
    

 