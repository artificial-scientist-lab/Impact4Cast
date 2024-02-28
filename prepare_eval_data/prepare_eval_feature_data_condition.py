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
from preprocess_utils import *
from features_utils import *
from train_model_utils import *
 


time_start_begin=time.time() 
logs_file_name='logs_eval_data_infos'   
store_folder="data_pair_solution"
pair_solution_data1=os.path.join(store_folder,"unconnected_2019_pair_solution_connected_2022_clean.parquet")


time_start = time.time()
full_eval_pair_result = pd.read_parquet(pair_solution_data1)
print(f"Done, combine all: {len(full_eval_pair_result)}; elapsed_time: {time.time() - time_start}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\nDone, combine all: {len(full_eval_pair_result)}; elapsed_time: {time.time() - time_start}")


day_origin = date(1990,1,1)
vertex_degree_cutoff=1
years_delta=3
min_edges=1
year_start=2022-years_delta

rnd_seed=42
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

edges_used=10**7
num_row = int(min(edges_used, len(full_eval_pair_result)))

time_start = time.time()
shuffled = full_eval_pair_result.sample(frac=1, random_state=rnd_seed)
eval_data_pair_solution = shuffled.head(num_row)

print(f"Done, eval_data_pair_solution: {len(eval_data_pair_solution)}; elapsed_time: {time.time() - time_start}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\nDone, eval_data_pair_solution: {len(eval_data_pair_solution)}; elapsed_time: {time.time() - time_start}")



store_eval_folder="data_eval" # store folder
if not os.path.exists(store_eval_folder):
    os.makedirs(store_eval_folder)
print(f"store files in {store_eval_folder}.....")

###----- store eval_data_pair_solution -----###     
time_start = time.time()
store_name=os.path.join(store_eval_folder,"eval_data_pair_solution_condition.parquet")
eval_data_pair_solution.to_parquet(store_name, compression='gzip')
print(f"eval_data_pair_solution: {len(eval_data_pair_solution)}; elapsed_time: {time.time() - time_start}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\neval_data_pair_solution: {len(eval_data_pair_solution)}; elapsed_time: {time.time() - time_start}")


###----- prepare features -----###
time_start = time.time()
data_folder="data_concept_graph" # folder that stores the full knowledge graph
graph_file=os.path.join(data_folder,"full_dynamic_graph.parquet") # load the full knowledge graph 
full_dynamic_graph = pd.read_parquet(graph_file)
print(f"{datetime.now()}: Done, read full_dynamic_graph: {len(full_dynamic_graph)}; elapsed_time: {time.time() - time_start}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, read full_dynamic_graph: {len(full_dynamic_graph)}; elapsed_time: {time.time() - time_start}")


start_time=time.time()
adj_mat_sparse=[]
node_neighbor_list=[]
num_neighbor_list=[]
for yy in [year_start,year_start-1,year_start-2]:
    data_file=os.path.join("data_for_features", f"adjacency_matrix_{yy}.gz")
    adj_mat=get_adjacency_matrix(full_dynamic_graph, yy, data_file)
    adj_mat_sparse.append(adj_mat)
    
    curr_node_neighbor=get_node_neighbor(adj_mat)
    node_neighbor_list.append(curr_node_neighbor)
    
    curr_num_neighbor = np.array(adj_mat.sum(axis=0)).flatten() # array 
    num_neighbor_list.append(curr_num_neighbor)
    
print(f"{datetime.now()}: Done, adjacency_matrix_sparse; elapsed_time: {time.time() - start_time}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, adjacency_matrix_sparse; elapsed_time: {time.time() - start_time}")
    

start_time=time.time()
vertex_features=get_all_node_feature(adj_mat_sparse, year_start, "data_for_features")
print(f"{datetime.now()}: Done, vertex_features; elapsed_time: {time.time() - start_time}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, vertex_features; elapsed_time: {time.time() - start_time}")
    

start_time=time.time()
vc_feature_list=[]
for yy in [year_start,year_start-1,year_start-2]:
    data_file=os.path.join("data_for_features", f"concept_node_citation_data_{yy}.parquet")
    vc_df=pd.read_parquet(data_file)
    vc_feature=vc_df.values
    vc_feature_list.append(vc_feature)
    
vertex_cfeatures=get_all_node_cfeature(vc_feature_list)
print(f"{datetime.now()}: Done, vertex_cfeatures; elapsed_time: {time.time() - start_time}") 
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, vertex_cfeatures; elapsed_time: {time.time() - start_time}")
    


time_start = time.time()
eval_pair_solution=eval_data_pair_solution.values
unconnected_vertex_pairs=eval_pair_solution[:,:2]

pair_features, pair_cfeatures=get_all_pair_features(vc_feature_list, node_neighbor_list, num_neighbor_list, unconnected_vertex_pairs, logs_file_name)

all_features=[vertex_features, vertex_cfeatures, pair_features, pair_cfeatures]

eval_data_features=get_all_feature(all_features, unconnected_vertex_pairs, logs_file_name)

print(f"finish; {len(eval_data_features)}; time: {time.time()-time_start}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\nfinish; {len(eval_data_features)}; time: {time.time()-time_start}")
    

###----- store eval_data_pair_feature -----### 
time_start = time.time()
store_name=os.path.join(store_eval_folder,"eval_data_pair_feature_condition.parquet")
data_eval_2022 = pd.DataFrame(eval_data_features)
 
# Convert column names to string
data_eval_2022.columns = data_eval_2022.columns.astype(str)
data_eval_2022.to_parquet(store_name, compression='gzip')  

print(f"store data_eval_2022: {len(data_eval_2022)}; elapsed_time: {time.time() - time_start}")
with open(logs_file_name+".txt", "a") as myfile:
    myfile.write(f"\nstore data_eval_2022: {len(data_eval_2022)}; elapsed_time: {time.time() - time_start}")
    myfile.write(f"\n\n{datetime.now()}: {time.time() - time_start_begin}")
    
