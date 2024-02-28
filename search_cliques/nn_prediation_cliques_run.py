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


parent_folder = f"{year_start}_cliques"
if not os.path.exists(parent_folder):
    os.mkdir(parent_folder)

log_folder = os.path.join(parent_folder, "log")
nn_output_folder = os.path.join(parent_folder, "nn_output")

try:
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    if not os.path.exists(nn_output_folder):
        os.makedirs(nn_output_folder)

except FileExistsError:
    pass

num=0  ## file num 0 -- 14, in total 14 parts, parallel computing

logs_run=os.path.join(log_folder,f"log_run{num}.txt")
with open(logs_run, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Start Run......") 

    
norm_factor_folder=f"{year_start}_norm_factor"
norm_file=os.path.join(norm_factor_folder,"Feature_Norm_Factor_File.gz") 
with gzip.open(norm_file, "rb") as f:
    data_max_feature, data_cmax_feature=pickle.load(f)
    
    
time_start = time.time()
train_data_folder = 'data_pair_solution'
train_pair_file=os.path.join(train_data_folder,"unconnected_pair_2022.parquet")
full_train_data_df = pd.read_parquet(train_pair_file)
full_train_data=full_train_data_df.values
full_train_data_df=pd.DataFrame()
with open(logs_run, "a") as myfile:
    myfile.write(f"\nDone, read unconnected_pair_2022: {len(full_train_data)}; {time.time() - time_start}s") 
    
    
    
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
        

vertex_features=get_all_node_feature(adj_mat_sparse, year_start, feature_folder)

start_time=time.time()
vc_feature_list=[]
for yy in [year_start,year_start-1,year_start-2]:
    data_file=os.path.join(feature_folder, f"concept_node_citation_data_{yy}.parquet")
    vc_df=pd.read_parquet(data_file)
    vc_feature=vc_df.values
    vc_feature_list.append(vc_feature)

vertex_cfeatures=get_all_node_cfeature(vc_feature_list)
 
with open(logs_run, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, adj_mat and vc_feature_list; elapsed_time: {time.time() - start_time}") 
    
    
    

split_type=0
num_class=2
out_norm=False

IR_num=[10]
# load 2019-2022, yy=year_start-years_delta
save_folders, train_log_folders=make_folders(year_start-years_delta, split_type, num_class, "train")
net_folder, loss_folder, figure_folder, result_folder= save_folders

IR_Str=format_IR(IR_num, split_type)
net_name=f"net_state_year_{year_start-years_delta}_delta_{years_delta}_class_{num_class}_{IR_Str}.pt"
net_file=os.path.join(net_folder,net_name)
logs_file_name=os.path.join(log_folder,f"nn_output_{year_start+years_delta}_"+IR_Str)

batch_size=1000 
lr_enc=3*10**-5
rnd_seed=42
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed) 
hyper_parameter=[batch_size, lr_enc, rnd_seed]
graph_parameter=[year_start,years_delta,vertex_degree_cutoff, min_edges]
user_parameter=[num_class, IR_num, split_type, out_norm]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 141
hidden_size = 600
if num_class<=2:
    output_size = 1  # Assuming n output classes
else:
    output_size = num_class

model_semnet = ff_network(input_size, hidden_size, output_size).to(device)

state_dict = torch.load(net_file, map_location=device)
model_semnet.load_state_dict(state_dict)


data_batch_size=1000
time_start=time.time()
data_size = full_train_data.shape[0]
num_files = 14
chunk_size = data_size // num_files  # This computes the average size

pair_start=num 
start_idx = pair_start * chunk_size
if pair_start == num_files - 1:
    end_idx = data_size
else:
    end_idx = start_idx + chunk_size

data_chunk = full_train_data[start_idx:end_idx]
full_train_data=None

with open(logs_run, "a") as myfile:
    myfile.write(f"\nrun chunk {start_idx} to {end_idx}......") 
    
logs_file_name=os.path.join(log_folder,f"NN_Prediction_File{pair_start}")

sub_chunk_size = int(5e6)  
sub_chunks = len(data_chunk) // sub_chunk_size
for j in range(sub_chunks):
    sub_start = j * sub_chunk_size
    if j == sub_chunks - 1:
        sub_end = len(data_chunk)
    else:
        sub_end = sub_start + sub_chunk_size
        
    pair_data=data_chunk[sub_start:sub_end]
 
    all_pair_features, all_pair_cfeatures=get_all_pair_features(vc_feature_list, node_neighbor_list, num_neighbor_list, pair_data, logs_file_name)
    node_pair_feature=[vertex_features, vertex_cfeatures, all_pair_features, all_pair_cfeatures]
    data_feature_sample=get_norm_features(node_pair_feature, data_max_feature, data_cmax_feature, pair_data, logs_file_name)
 
    with torch.no_grad():
        predictions=eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_sample, user_parameter)
        nn_out=os.path.join(nn_output_folder,f"NN_Prediction_File{pair_start}_P{j}.gz")
        
        with gzip.open(nn_out, 'wb') as output_file:
            pickle.dump(predictions, output_file)
 
    #print(f"index: {j}; {sub_start} to {sub_chunks}; time: {time.time()-start_time}") 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nindex: {j}; {sub_start} to {sub_chunks}; time: {time.time()-start_time}") 
        
    start_time=time.time()
    

with open(logs_run, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Finish Run......") 