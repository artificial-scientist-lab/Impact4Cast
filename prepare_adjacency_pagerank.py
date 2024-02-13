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

NUM_OF_VERTICES=37960   ## number of vertices in the graph

def get_adjacency_matrix(full_graph, year, store_folder):
    
    start_time=time.time()
    data_file=os.path.join(store_folder, f"adjacency_matrix_{year}.gz")
    
    if os.path.exists(data_file):
        with gzip.open(data_file, "rb") as f:
            adjacency_matrix=pickle.load(f)
        print(f"{datetime.now()}: Done, read adjacency_matrix; elapsed_time: {time.time() - start_time}")    
    else:
        
        day_origin = date(1990,1,1)
        day_curr=(date(year,12,31)- day_origin).days
        full_graph_edges=full_graph[full_graph['time']<=day_curr]

        all_graph_edges=full_graph_edges.values
        adjacency_matrix = sparse.csr_matrix((np.ones(len(all_graph_edges), dtype=np.uint64), (all_graph_edges[:,0], all_graph_edges[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
        adjacency_matrix= adjacency_matrix + adjacency_matrix.transpose()
        adjacency_matrix = (adjacency_matrix > 0).astype(int) 

        with gzip.open(data_file, "wb") as f:
            pickle.dump(adjacency_matrix, f)
            
        print(f"Done year: {year}; num of nodes: {adjacency_matrix.shape[0]}; num of edges: {adjacency_matrix.sum()/2}; {time.time() - start_time}s")
        with open("log_adjacent_pagerank.txt", "a") as myfile:
            myfile.write(f"\nDone year: {year}; num of nodes: {adjacency_matrix.shape[0]}; num of edges: {adjacency_matrix.sum()/2}; {time.time() - start_time}s")

    return adjacency_matrix



def get_pagerank_score(adjacency_matrix, year, store_folder):
    
    data_file=os.path.join(store_folder,f"pagerank_score_{year}.gz")
    print(f"{datetime.now()}: start loading pagerank_score files")
    if os.path.exists(data_file):
        start_time=time.time()
        with gzip.open(data_file, "rb") as f:
            pagerank_score=pickle.load(f)  
        print(f"\n{datetime.now()}: Done, loading pagerank_score; {time.time() - start_time}s")

    else: ## 5-6min
        start_time=time.time()
        graph=nx.from_scipy_sparse_array(adjacency_matrix)
       
        with open("log_adjacent_pagerank.txt", "a") as myfile:
            myfile.write(f"\ndone graph ...{time.time()-start_time}s")
        
        start_time=time.time()
        pagerank = nx.algorithms.link_analysis.pagerank(graph)
    
        with open("log_adjacent_pagerank.txt", "a") as myfile:
            myfile.write(f"\ndone pagerank; {time.time()-start_time}s")
            
        start_time=time.time()
        pagerank_score = np.zeros(shape=(NUM_OF_VERTICES,), dtype=np.float32)
        for i in range(NUM_OF_VERTICES):
            pagerank_score[i] = pagerank[i]

        with open("log_adjacent_pagerank.txt", "a") as myfile:
            myfile.write(f"\nDone, pagerank_score: {len(pagerank_score)}; {time.time() - start_time}s")
            
        start_time=time.time()
        with gzip.open(data_file, "wb") as f:
            pickle.dump(pagerank_score, f)

        with open("log_adjacent_pagerank.txt", "a") as myfile:
            myfile.write(f"\n{datetime.now()}: Store pagerank_score files; {time.time() - start_time}s")
        
    return pagerank_score



time_start = time.time()
data_folder="data_concept_graph"

# Read all concepts together with time, citation information
graph_file=os.path.join(data_folder,"full_dynamic_graph.parquet")
full_edge_dynamic_data = pd.read_parquet(graph_file)

print(f"Done, elapsed_time: {time.time() - time_start}\n full_edge_dynamic_data: {len(full_edge_dynamic_data)};\n")


log_files="log_adjacent_pagerank.txt"

data_folder="data_for_features"
years=[2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

start_time1=time.time()
for yy in years:
    
    print(f"{datetime.now()}: start adjacency_matrix")
    with open(log_files, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: start adjacency_matrix")
        
    adjacency_matrix_sparse=get_adjacency_matrix(full_edge_dynamic_data, yy, data_folder)
    print(f"{datetime.now()}: finish adjacency_matrix")
    with open(log_files, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: finish adjacency_matrix")
        
    pagerank_score=get_pagerank_score(adjacency_matrix_sparse, yy, data_folder)
    print(f"{datetime.now()}: finish pagerank_score")
    print(f"done, year {yy}: {time.time() - start_time1}s")
    with open(log_files, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: done, year {yy}: {time.time() - start_time1}s")
    start_time1=time.time()

