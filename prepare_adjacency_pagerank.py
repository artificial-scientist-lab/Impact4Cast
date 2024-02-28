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
from features_utils import get_adjacency_matrix, get_pagerank_score

NUM_OF_VERTICES=37960   ## number of vertices in the graph

time_start = time.time()
data_folder="data_concept_graph" # the folder which stores the full dynamic knowledge graph

# Read all concepts together with time, citation information
graph_file=os.path.join(data_folder,"full_dynamic_graph.parquet")
full_edge_dynamic_data = pd.read_parquet(graph_file)

print(f"Done, elapsed_time: {time.time() - time_start}\n full_edge_dynamic_data: {len(full_edge_dynamic_data)};\n")

log_files="log_adjacent_pagerank.txt" # just for logging the running situation

data_folder="data_for_features" # folder to store the generated adjacency_matrix files and pagerank files for different years
years=[2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

start_time1=time.time()
for yy in years:
    
    print(f"{datetime.now()}: start adjacency_matrix")
    with open(log_files, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: start adjacency_matrix")

    data_file=os.path.join(data_folder, f"adjacency_matrix_{yy}.gz")   
    adjacency_matrix_sparse=get_adjacency_matrix(full_edge_dynamic_data, yy, data_file)
    print(f"{datetime.now()}: finish adjacency_matrix")
    with open(log_files, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: finish adjacency_matrix")
        
    data_file=os.path.join(data_folder,f"pagerank_score_{yy}.gz")
    pagerank_score=get_pagerank_score(adjacency_matrix_sparse, data_file)
    print(f"{datetime.now()}: finish pagerank_score")
    print(f"done, year {yy}: {time.time() - start_time1}s")
    with open(log_files, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: done, year {yy}: {time.time() - start_time1}s")
    start_time1=time.time()

