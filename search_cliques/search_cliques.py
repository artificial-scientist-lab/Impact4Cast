import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import networkx as nx
import pandas as pd
from collections import defaultdict,Counter
from datetime import datetime, date
from itertools import combinations
from general_utils import *

def add_edge_to_graph(graph, edge):
    u, v = edge
    graph.setdefault(u, set()).add(v)
    graph.setdefault(v, set()).add(u)

def is_clique(graph, nodes):
    for i, node in enumerate(nodes):
        for neighbor in nodes[i+1:]:
            if neighbor not in graph[node]:
                return False
    return True

def expand_clique(graph, current_clique, desired_size):
    if len(current_clique) == desired_size:
        return current_clique

    last_node = current_clique[-1]
    possible_nodes = graph[last_node] - set(current_clique)

    if len(current_clique) + len(possible_nodes) < desired_size:
        return None

    for neighbor in possible_nodes:
        # More optimal way to check if all nodes are connected to the neighbor
        for node in current_clique:
            if neighbor not in graph[node]:
                break
        else: # Only enters this block if the loop didn't break
            result = expand_clique(graph, current_clique + [neighbor], desired_size)
            if result:
                return result

    return None

## initial settings
day_origin = date(1990,1,1)
vertex_degree_cutoff=1
years_delta=3
min_edges=1
year_start=2025-years_delta

parent_folder = f"{year_start}_cliques"
nn_output_folder = os.path.join(parent_folder, "nn_output")

split_type=0
num_class=2
out_norm=False
IR_num=[10]
IR_Str=format_IR(IR_num, split_type)
store_clique_file=os.path.join(parent_folder,f"clique_result_{IR_Str}.txt")

logs_file_name=os.path.join(parent_folder,f"clique_log_{IR_Str}.txt")
with open(logs_file_name, 'a') as f:
    f.write(f"\n{datetime.now()}: Start....\n")
    
    
## load predictions from nn for all unconnected pairs  
num_max=14
subnum_max=9
out_index=list(range(num_max)) 
sub_index= list(range(subnum_max))

time_start=time.time()
flattened_arrays=[]
for id_out in out_index:
    for id_in in sub_index:
        norm_file=os.path.join(nn_output_folder,f"NN_Prediction_File{id_out}_P{id_in}.gz")
        with gzip.open(norm_file, "rb") as f:
            nn_predictions=pickle.load(f)
            
        flattened_arrays.append(nn_predictions.flatten())
        
all_predictions = np.concatenate(flattened_arrays)
with open(logs_file_name, 'a') as f:
    f.write(f"\nfinish loading prediction from NN; {time.time()-time_start}")
    
    
## load unconnected pairs     
train_data_folder = 'data_pair_solution'
train_pair_file=os.path.join(train_data_folder,"unconnected_pair_2022.parquet")

time_start = time.time()
full_train_data_df = pd.read_parquet(train_pair_file)
full_train_data=full_train_data_df.values
full_train_data_df=pd.DataFrame()
print(f"Done, read unconnected_pairs: {len(full_train_data)}; elapsed_time: {time.time() - time_start}")
with open(logs_file_name, 'a') as f:
    f.write(f"\nDone, read unconnected_pairs: {len(full_train_data)}; elapsed_time: {time.time() - time_start}")
    

## sorting 
time_start=time.time()
sorted_indices = np.argsort(all_predictions)[::-1]
full_train_data_sorted = full_train_data[sorted_indices]
with open(logs_file_name, 'a') as f:
    f.write(f"\nDone, sort: {len(full_train_data_sorted)}; elapsed_time: {time.time() - time_start}")
    

## read concepts
concept_folder="data_concept_graph"
concepts_files = os.path.join(concept_folder, 'full_concepts_for_openalex.txt')
with open(concepts_files, 'r') as file:
    full_concepts = [concept.strip() for concept in file.readlines()]


## search cliques

        
graph = {}
desired_size = 2
start_time = time.time()
start_time1 = time.time()

for idx, edge in enumerate(full_train_data_sorted):
    if idx % 10000 == 0 and idx!=0:
        print(f"process {idx}/{len(full_train_data_sorted)}; {time.time()-start_time1}")
        with open(logs_file_name, 'a') as f:
            f.write(f"\n     process {idx}/{len(full_train_data_sorted)}; {time.time()-start_time1}")
        start_time1 = time.time()
        
        
    add_edge_to_graph(graph, tuple(edge))
    
    if desired_size == 2:
        clique = edge.tolist()
    else:
        # Start the search using the current edge
        clique = None
        for starting_node in edge:
            clique = expand_clique(graph, [starting_node], desired_size)
            if clique:
                break
                
    if clique:
        #print(f"Found clique of size {desired_size}: {set(clique)}\n")
        clique_concepts = [full_concepts[i] for i in clique] 
        with open(store_clique_file, 'a') as f:
            f.write(f"\nEdge_Num={idx+1}; Clique_Size={desired_size}: {clique}")
            f.write(f"\nConcept: {clique_concepts}\n")
        
        with open(logs_file_name, 'a') as f:
            f.write(f"\nidx: {idx}: {idx/len(full_train_data_sorted)}; Finish clique of size {desired_size}; {time.time()-start_time}")
        desired_size += 1
        start_time = time.time()
 
         
with open(logs_file_name, 'a') as f:
    f.write(f"\n{datetime.now()}: Finish all....")