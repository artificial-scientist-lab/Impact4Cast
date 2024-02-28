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
from general_utils import *
from features_utils import *
from preprocess_utils import *
from train_model_utils import * 


day_origin = date(1990,1,1)
vertex_degree_cutoff=1
years_delta=3
min_edges=1
year_start=2025-years_delta

rnd_seed=42
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)  


parent_folder = f"{year_start}_norm_factor"
if not os.path.exists(parent_folder):
    os.mkdir(parent_folder)

log_folder = os.path.join(parent_folder, "log")
norm_feature_folder = os.path.join(parent_folder, "max_feature")

try:
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)
    
    if not os.path.exists(norm_feature_folder):
        os.makedirs(norm_feature_folder)

except FileExistsError:
    pass

num=0  ## file num 0 -- 14, in total 14 parts, parallel computing
logs_run=os.path.join(log_folder,f"log_run{num}.txt")
with open(logs_run, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Start Run......") 


feature_folder="data_for_features"
start_time=time.time()
adj_mat_sparse=[]
node_neighbor_list=[]
num_neighbor_list=[]
for yy in [year_start,year_start-1,year_start-2]:
    data_file=os.path.join(feature_folder, f"adjacency_matrix_{yy}.gz")
    if os.path.exists(data_file):
        with gzip.open(data_file, "rb") as f:
            adj_mat=pickle.load(f)
        
        adj_mat_sparse.append(adj_mat)
        curr_node_neighbor=get_node_neighbor(adj_mat)
        node_neighbor_list.append(curr_node_neighbor)

        curr_num_neighbor = np.array(adj_mat.sum(axis=0)).flatten() # array 
        num_neighbor_list.append(curr_num_neighbor)
    
    else:
        print("no such file...., please create it using get_adjacency_matrix function")
        
#print(f"{datetime.now()}: Done, adjacency_matrix_sparse; elapsed_time: {time.time() - start_time}")
vc_feature_list=[]
for yy in [year_start,year_start-1,year_start-2]:
    data_file=os.path.join(feature_folder, f"concept_node_citation_data_{yy}.parquet")
    vc_df=pd.read_parquet(data_file)
    vc_feature=vc_df.values
    vc_feature_list.append(vc_feature)
    
with open(logs_run, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, adj_mat and vc_feature_list; elapsed_time: {time.time() - start_time}") 
    
    
train_data_folder = 'data_pair_solution'
train_pair_file=os.path.join(train_data_folder,"unconnected_pair_2022.parquet")

time_start = time.time()
full_train_data_df = pd.read_parquet(train_pair_file)
full_train_data=full_train_data_df.values
full_train_data_df=pd.DataFrame()

#print(f"Done, read unconnected_pair_2022: {len(full_train_data)}; elapsed_time: {time.time() - time_start}")
with open(logs_run, "a") as myfile:
    myfile.write(f"\nDone, read unconnected_pair_2022: {len(full_train_data)}; {time.time() - time_start}s") 
    

data_size = full_train_data.shape[0]
num_files = 14
chunk_size = data_size // num_files  # This computes the average size

pair_start=0 # max 13
start_idx = pair_start * chunk_size
if pair_start == num_files - 1:
    end_idx = data_size
else:
    end_idx = start_idx + chunk_size

data_chunk = full_train_data[start_idx:end_idx]
full_train_data=None

#print(f"run chunk {start_idx} to {end_idx}; {time.time()-start_time}")
with open(logs_run, "a") as myfile:
    myfile.write(f"run chunk {start_idx} to {end_idx}......") 
    
    
logs_file_name=os.path.join(log_folder,f"Feature_MaxNum_File{pair_start}")

sub_chunk_size = int(5e6)  
sub_chunks = len(data_chunk) // sub_chunk_size

start_time=time.time()
for j in range(sub_chunks):
    sub_start = j * sub_chunk_size
    if j == sub_chunks - 1:
        sub_end = len(data_chunk)
    else:
        sub_end = sub_start + sub_chunk_size

    sub_data_chunk = data_chunk[sub_start:sub_end]
    all_pair_features, all_pair_cfeatures=get_all_pair_features(vc_feature_list, node_neighbor_list, num_neighbor_list, sub_data_chunk, logs_file_name)
    
    max_values0=return_col_max(all_pair_features[0])
    max_values1=return_col_max(all_pair_features[1])
    max_values2=return_col_max(all_pair_features[2])

    cmax_values0=return_col_max(all_pair_cfeatures[0])
    cmax_values1=return_col_max(all_pair_cfeatures[1])
    cmax_values2=return_col_max(all_pair_cfeatures[2])

    data_max_feature=[max_values0,max_values1,max_values2]
    data_cmax_feature=[cmax_values0,cmax_values1,cmax_values2]
    
    norm_feature_file=os.path.join(norm_feature_folder,f"Feature_MaxNum_File{pair_start}_P{j}.gz")
    with gzip.open(norm_feature_file, 'wb') as output_file:
        pickle.dump([data_max_feature, data_cmax_feature], output_file)
    
    #print(f"index: {j}; {sub_start} to {sub_chunks}; time: {time.time()-start_time}") 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nindex: {j}; {sub_start} to {sub_chunks}; time: {time.time()-start_time}") 
        
    start_time=time.time()
    

with open(logs_run, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Finish Run......") 