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

def get_adjacency_matrix(full_graph, year, data_file):
    """
    Prepare the adjacency matrix of a knowledge graph up to year (set as y-12-31)
    
    full_graph: the full knowledge graph stored in pandas
    year: cut-off year, set as date(year,12,31)
    data_file: store the adjacency matrix
    """ 

    start_time=time.time()
    if os.path.exists(data_file):
        with gzip.open(data_file, "rb") as f:
            adjacency_matrix=pickle.load(f)
        print(f"{datetime.now()}: Done {year}, read adjacency_matrix; {time.time() - start_time}s")    
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

    return adjacency_matrix



def get_pagerank_score(adjacency_matrix, data_file):
    
    print(f"{datetime.now()}: getting the pagerank score")
    if os.path.exists(data_file):
        start_time=time.time()
        with gzip.open(data_file, "rb") as f:
            pagerank_score=pickle.load(f)  
        print(f"{datetime.now()}: Done, loading pagerank_score; {time.time() - start_time}s")

    else: ## roughly 5-6mins
        start_time=time.time()
        graph=nx.from_scipy_sparse_array(adjacency_matrix)
        
        pagerank = nx.algorithms.link_analysis.pagerank(graph)
         
        pagerank_score = np.zeros(shape=(NUM_OF_VERTICES,), dtype=np.float32)
        for i in range(NUM_OF_VERTICES):
            pagerank_score[i] = pagerank[i]

        with gzip.open(data_file, "wb") as f:
            pickle.dump(pagerank_score, f)
        print(f"{datetime.now()}: done pagerank_score; {time.time() - start_time}s")
        
    return pagerank_score


##################################################################################

# get the connected neighbors for each node, list type; node_neighbors[0] is numpy array
def get_node_neighbor(adjacency_matrix: sparse.csr_matrix): 
    return [adjacency_matrix.getrow(i).indices for i in range(NUM_OF_VERTICES)] #adjacency_matrix.shape[0]


# get the number of connected neighbors for each node 
def get_num_neighbor(adjacency_matrix: sparse.csr_matrix):
    
    num_neighbor = np.array(adjacency_matrix.sum(axis=0)).flatten() # array 
    #rank_num_neighbor=rankdata(num_neighbor)
 
    return num_neighbor


# get the num of shared neighbors for one vertex pair
def get_num_shared_neighbor(node_neighbor, vertex_pairs):
    num_shared_neighbor = np.zeros(len(vertex_pairs), dtype=int)
    for id_x, curr_v in enumerate(vertex_pairs):
        v1 = int(curr_v[0])
        v2 = int(curr_v[1])
        curr_common_neighbor = np.intersect1d(node_neighbor[v1], node_neighbor[v2]).size
        num_shared_neighbor[id_x] = curr_common_neighbor
    return num_shared_neighbor
        

# get the features for each node, year gap=3 
def get_all_node_feature(adjacency_matrix_list, year, data_folder):
    
    adjacency_matrix0, adjacency_matrix1, adjacency_matrix2 = adjacency_matrix_list
    
    num_neighbors0 = get_num_neighbor(adjacency_matrix0)
    num_neighbors1 = get_num_neighbor(adjacency_matrix1)
    num_neighbors2 = get_num_neighbor(adjacency_matrix2)
    
    num_diff_1_year = num_neighbors0 - num_neighbors1
    num_diff_2_year = num_neighbors0 - num_neighbors2
    ranknum_diff_1_year = rankdata(num_diff_1_year)
    ranknum_diff_2_year = rankdata(num_diff_2_year)
    
    data_file=os.path.join(data_folder,f"pagerank_score_{year}.gz") 
    pagerank_s0 = get_pagerank_score(adjacency_matrix0, data_file)

    data_file=os.path.join(data_folder,f"pagerank_score_{year-1}.gz") 
    pagerank_s1 = get_pagerank_score(adjacency_matrix1, data_file)

    data_file=os.path.join(data_folder,f"pagerank_score_{year-2}.gz") 
    pagerank_s2 = get_pagerank_score(adjacency_matrix2, data_file)

    # Collecting all arrays in a list and stacking them at the end
    all_features = [num_neighbors0, num_neighbors1, num_neighbors2,
                    num_diff_1_year, num_diff_2_year,
                    ranknum_diff_1_year, ranknum_diff_2_year,
                    pagerank_s0, pagerank_s1, pagerank_s2]

    
    node_features = np.vstack(all_features)
    
    return node_features


# get the features for each vertex pair
def get_pair_feature(node_neighbor: list, num_neighbor: np.ndarray, vertex_list: np.ndarray): 
    
    num_pairs = len(vertex_list)
    
    # Pre-allocate the result array
    num_features=7
    pair_features = np.zeros((num_pairs, num_features))
    
    for id_v, curr_v in enumerate(vertex_list):
        v1 = int(curr_v[0])
        v2 = int(curr_v[1])
        
        num_shared_neighbor = np.intersect1d(node_neighbor[v1], node_neighbor[v2]).size
        n_v1 = num_neighbor[v1]
        n_v2 = num_neighbor[v2]
        
        if n_v1 == 0 or n_v2 == 0:
            gem_index=0  # geometric_index
            cos_index=0  # cosine_index
            sps_index=0  # simpson index 
            pre_index=0  # preferential_attachment
            
        else:
            gem_index = num_shared_neighbor**2 / (n_v1 * n_v2)
            cos_index = gem_index**0.5
            sps_index = num_shared_neighbor / np.min([n_v1, n_v2])
            pre_index = n_v1 * n_v2
            
        if n_v1 == 0 and n_v2 == 0:
            sod_index=0  # Sørensen–Dice coefficient
        else:
            sod_index = 2*num_shared_neighbor / (n_v1 + n_v2)
            
        if n_v1 + n_v2 - num_shared_neighbor>0:
            jac_index = num_shared_neighbor/(n_v1 + n_v2 - num_shared_neighbor)            
        else:
            jac_index=0   
        
        pair_features[id_v] = [num_shared_neighbor, gem_index, cos_index, sps_index, pre_index, sod_index, jac_index]
        # some testing
        #xxx=[num_shared_neighbor, gem_index, cos_index, sps_index, pre_index, sod_index, jac_index]
    #print(f"xxx: {len(xxx)}; num_pair_feature={num_features}")
    return pair_features


##################################################################################
# get the citation feature for each node
def get_all_node_cfeature(node_cfeature_list):
    
    node_cfeature0, node_cfeature1, node_cfeature2 = node_cfeature_list
    
    all_features = []
    
    # 1: c2016 2: ct_2016 3: ct_delta 4: num 5: c2016_m 6: ct_2016_m 7: ct_delta_m
    indices_to_process = list(range(1,node_cfeature0.shape[1]))
    for index in indices_to_process:
        # Extract columns from each numpy array
        feature0 = node_cfeature0[:, index]
        feature1 = node_cfeature1[:, index]
        feature2 = node_cfeature2[:, index]
        
        all_features.extend([feature0, feature1, feature2])
        # index==4:
            #all_features.extend([feature0, feature1, feature2, rankdata(feature0), rankdata(feature1), rankdata(feature2)])
        #else:
            #all_features.extend([feature0, feature1, feature2])
            
    # Compute differences for specific features and their ranks
    diff_features = [2, 4] 
    for index in diff_features:
        feature0 = node_cfeature0[:, index]
        feature1 = node_cfeature1[:, index]
        feature2 = node_cfeature2[:, index]

        diff_1_year = feature0 - feature1
        diff_2_year = feature0 - feature2
        
        all_features.extend([diff_1_year, diff_2_year, rankdata(diff_1_year), rankdata(diff_2_year)])
    
    # Stack all features at once
    node_cfeatures = np.vstack(all_features)
    #print(f"node_cfeatures: {node_cfeatures.shape}")
    
    return node_cfeatures


def get_pair_cfeature(data_cparameters, vertex_list):
     
    # 1: c2016 2: ct_2016 3: ct_delta 4: num 5: c2016_m 6: ct_2016_m 7: ct_delta_m
    curr_num_c, num_total_c, num_delta_c, num_cdegree, curr_num_cm, num_total_cm, num_delta_cm=data_cparameters
    
    # Pre-allocate memory for the resulting array. 
    num_pair_feature=14
    pair_cfeatures = np.zeros((len(vertex_list), num_pair_feature))
    
    for id_v, curr_v in enumerate(vertex_list):
        v1, v2 = int(curr_v[0]), int(curr_v[1])
        features = []
        if num_cdegree[v1] or num_cdegree[v2]:
            features.extend([(curr_num_c[v1] + curr_num_c[v2]) / (num_cdegree[v1] + num_cdegree[v2]),
                             (curr_num_c[v1] * curr_num_c[v2]) / (num_cdegree[v1] + num_cdegree[v2])])
        else:
            features.extend([0, 0])

        features.extend([curr_num_cm[v1] + curr_num_cm[v2],
                         num_total_cm[v1] + num_total_cm[v2],
                         num_delta_c[v1] + num_delta_c[v2],
                         num_delta_cm[v1] + num_delta_cm[v2]])
        
        features.extend([min(curr_num_c[v1], curr_num_c[v2]),
                         max(curr_num_c[v1], curr_num_c[v2]),
                         min(num_total_c[v1], num_total_c[v2]),
                         max(num_total_c[v1], num_total_c[v2]),
                         min(num_delta_c[v1], num_delta_c[v2]),
                         max(num_delta_c[v1], num_delta_c[v2]),
                         min(num_cdegree[v1], num_cdegree[v2]),
                         max(num_cdegree[v1], num_cdegree[v2]),
                        ])

        # Assign the computed features directly to the pre-allocated array
        pair_cfeatures[id_v] = features
        #xxx=features
    
    #print(f"xxx: {len(features)}; num_pair_feature={num_pair_feature}")
    return pair_cfeatures


###############################################

def rescaling_col(features: np.ndarray):
    
    max_values = features.max(axis=0, keepdims=True)
    max_values = np.where(max_values == 0, 1, max_values) ## if the max is zero, then do not divide max
    normalized_arr = features / max_values
 
    return normalized_arr

def rescaling_row(features: np.ndarray):
    
    max_values = features.max(axis=1, keepdims=True)
    max_values = np.where(max_values == 0, 1, max_values)
    normalized_arr = features / max_values
 
    return normalized_arr

def return_col_max(features: np.ndarray):
    max_values = features.max(axis=0, keepdims=True)
    max_values = np.where(max_values == 0, 1, max_values) ## if the max is zero, then do not divide max
    return max_values

def get_all_pair_features(node_cfeature_list, node_neighbor_list, num_neighbor_list, vertex_list, logs_file_name):
    
    node_c0, node_c1, node_c2 =node_cfeature_list
    node_neighbor0, node_neighbor1, node_neighbor2 =node_neighbor_list
    num_neighbor0, num_neighbor1, num_neighbor2=num_neighbor_list

    #print(f"{datetime.now()}: start extract_features")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: start extract_features")
    
    start_time=time.time()
    pair_feature0=get_pair_feature(node_neighbor0, num_neighbor0, vertex_list)
    #print(f"Finish pair_feature0, {len(pair_feature0)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_feature0, {len(pair_feature0)}; time: {time.time()-start_time}")
        
    start_time=time.time()
    pair_feature1=get_pair_feature(node_neighbor1, num_neighbor1, vertex_list)
    #print(f"Finish pair_feature1, {len(pair_feature1)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_feature1, {len(pair_feature1)}; time: {time.time()-start_time}")
        
        
    start_time=time.time()
    pair_feature2=get_pair_feature(node_neighbor2, num_neighbor2, vertex_list)
    #print(f"Finish pair_feature2, {len(pair_feature2)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_feature2, {len(pair_feature2)}; time: {time.time()-start_time}")
        

    start_time=time.time() 
    node_cparameters = [node_c0[:, i] for i in range(1, node_c0.shape[1])]
    pair_cfeature0=get_pair_cfeature(node_cparameters, vertex_list)
    #print(f"Finish pair_cfeature0, {len(pair_cfeature0)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_cfeature0, {len(pair_cfeature0)}; time: {time.time()-start_time}")
        
    start_time=time.time()     
    node_cparameters = [node_c1[:, i] for i in range(1, node_c1.shape[1])]
    pair_cfeature1=get_pair_cfeature(node_cparameters, vertex_list)
    #print(f"Finish pair_cfeature1, {len(pair_cfeature1)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_cfeature1, {len(pair_cfeature1)}; time: {time.time()-start_time}")
      
    start_time=time.time() 
    node_cparameters = [node_c2[:, i] for i in range(1, node_c2.shape[1])]
    pair_cfeature2=get_pair_cfeature(node_cparameters, vertex_list)
    #print(f"Finish pair_cfeature2, {len(pair_cfeature2)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_cfeature2, {len(pair_cfeature2)}; time: {time.time()-start_time}")
            
    all_pair_feature=[pair_feature0, pair_feature1, pair_feature2]
    all_pair_cfeature=[pair_cfeature0, pair_cfeature1, pair_cfeature2]
    
    return all_pair_feature, all_pair_cfeature
    

def get_pure_pair_features(node_neighbor_list, num_neighbor_list, vertex_list, logs_file_name):
    
     
    node_neighbor0, node_neighbor1, node_neighbor2 =node_neighbor_list
    num_neighbor0, num_neighbor1, num_neighbor2=num_neighbor_list

    #print(f"{datetime.now()}: start extract_features")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\n{datetime.now()}: start extract_features")
    
    start_time=time.time()
    pair_feature0=get_pair_feature(node_neighbor0, num_neighbor0, vertex_list)
    #print(f"Finish pair_feature0, {len(pair_feature0)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_feature0, {len(pair_feature0)}; time: {time.time()-start_time}")
        
    start_time=time.time()
    pair_feature1=get_pair_feature(node_neighbor1, num_neighbor1, vertex_list)
    #print(f"Finish pair_feature1, {len(pair_feature1)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_feature1, {len(pair_feature1)}; time: {time.time()-start_time}")
        
        
    start_time=time.time()
    pair_feature2=get_pair_feature(node_neighbor2, num_neighbor2, vertex_list)
    #print(f"Finish pair_feature2, {len(pair_feature2)}; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish pair_feature2, {len(pair_feature2)}; time: {time.time()-start_time}")
              
    all_pair_feature=[pair_feature0, pair_feature1, pair_feature2]
    
    return all_pair_feature


        
def get_all_feature(node_pair_features, vertex_list, logs_file_name):
    
    node_feature, node_cfeature, pair_feature, pair_cfeature = node_pair_features
    pair_feature0, pair_feature1, pair_feature2 = pair_feature
    pair_cfeature0, pair_cfeature1, pair_cfeature2 = pair_cfeature
     
    start_time=time.time()
    norm_node_feature=rescaling_row(node_feature)
    norm_node_cfeature=rescaling_row(node_cfeature)
    
    norm_pair_feature0=rescaling_col(pair_feature0)
    norm_pair_feature1=rescaling_col(pair_feature1)
    norm_pair_feature2=rescaling_col(pair_feature2)
    norm_pair_cfeature0=rescaling_col(pair_cfeature0)
    norm_pair_cfeature1=rescaling_col(pair_cfeature1)
    norm_pair_cfeature2=rescaling_col(pair_cfeature2)    
    print(f"Finish rescaling; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish rescaling; time: {time.time()-start_time}")
        
    start_time=time.time()
    num_features = 2*len(norm_node_feature)+2*len(norm_node_cfeature)+3*len(norm_pair_feature0[0])+3*len(norm_pair_cfeature0[0])
    store_features = np.zeros((len(vertex_list), num_features))
    
    print(f"shape: {norm_node_feature.shape}; {norm_node_cfeature.shape}; {norm_pair_feature0.shape}; {norm_pair_cfeature0.shape}")
    print(f"store_features: {store_features.shape}")
    for id_v, curr_v in enumerate(vertex_list):

        vals=[]
        v1, v2 = int(curr_v[0]), int(curr_v[1])
        
        for ii in range(len(norm_node_feature)):
            vals.append(norm_node_feature[ii][v1])
            vals.append(norm_node_feature[ii][v2])

        for ii in range(len(norm_node_cfeature)):
            vals.append(norm_node_cfeature[ii][v1])
            vals.append(norm_node_cfeature[ii][v2]) 
             
        for ii in range(len(norm_pair_feature0[0])):
            vals.append(norm_pair_feature0[:,ii][id_v])
            vals.append(norm_pair_feature1[:,ii][id_v])
            vals.append(norm_pair_feature2[:,ii][id_v])
 
        for ii in range(len(norm_pair_cfeature0[0])):
            vals.append(norm_pair_cfeature0[:,ii][id_v])
            vals.append(norm_pair_cfeature1[:,ii][id_v])
            vals.append(norm_pair_cfeature2[:,ii][id_v])

        #store_features.append(vals)  # just in case [[]] not []
        store_features[id_v] = vals

        if id_v%10**5==0: #if ii%10**4==0:

            print(f'    compute_all_properties_of_list progress: ({time.time()-start_time} sec), {id_v/10**6}M/{len(vertex_list)/10**6}M')
            with open(logs_file_name+"_logs.txt", "a") as myfile:
                myfile.write(f'\n    compute_all_properties_of_list progress: ({time.time()-start_time} sec), {id_v/10**6}M/{len(vertex_list)/10**6}M')
            start_time=time.time()

    print('Finish store_features') 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write('\nFinish store_features')
    
    return store_features


def get_all_pure_feature(node_pair_features, vertex_list, logs_file_name):
    
    node_feature, pair_feature = node_pair_features
    pair_feature0, pair_feature1, pair_feature2 = pair_feature

     
    start_time=time.time()
    norm_node_feature=rescaling_row(node_feature)

    norm_pair_feature0=rescaling_col(pair_feature0)
    norm_pair_feature1=rescaling_col(pair_feature1)
    norm_pair_feature2=rescaling_col(pair_feature2)
  
    print(f"Finish rescaling; time: {time.time()-start_time}")
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write(f"\nFinish rescaling; time: {time.time()-start_time}")
        
    start_time=time.time()
    num_features = 2*len(norm_node_feature)+3*len(norm_pair_feature0[0])
    store_features = np.zeros((len(vertex_list), num_features))
    
    print(f"shape: {norm_node_feature.shape}; {norm_pair_feature0.shape}")
    print(f"store_features: {store_features.shape}")
    for id_v, curr_v in enumerate(vertex_list):

        vals=[]
        v1, v2 = int(curr_v[0]), int(curr_v[1])
        
        for ii in range(len(norm_node_feature)):
            vals.append(norm_node_feature[ii][v1])
            vals.append(norm_node_feature[ii][v2])
             
        for ii in range(len(norm_pair_feature0[0])):
            vals.append(norm_pair_feature0[:,ii][id_v])
            vals.append(norm_pair_feature1[:,ii][id_v])
            vals.append(norm_pair_feature2[:,ii][id_v])

        #store_features.append(vals)  # just in case [[]] not []
        store_features[id_v] = vals

        if id_v%10**5==0: #if ii%10**4==0:

            print(f'    compute_all_properties_of_list progress: ({time.time()-start_time} sec), {id_v/10**6}M/{len(vertex_list)/10**6}M')
            with open(logs_file_name+"_logs.txt", "a") as myfile:
                myfile.write(f'\n    compute_all_properties_of_list progress: ({time.time()-start_time} sec), {id_v/10**6}M/{len(vertex_list)/10**6}M')
            start_time=time.time()

    print('Finish store_features') 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write('\nFinish store_features')
    
    return store_features
    
        
def get_norm_features(node_pair_features, data_max_fature, data_cmax_fature, vertex_list, logs_file_name):
    
    node_feature, node_cfeature, pair_feature, pair_cfeature = node_pair_features
    pair_feature0, pair_feature1, pair_feature2 = pair_feature
    pair_cfeature0, pair_cfeature1, pair_cfeature2 = pair_cfeature
     
    max_values0,max_values1,max_values2=data_max_fature
    cmax_values0,cmax_values1,cmax_values2=data_cmax_fature

    norm_node_feature=rescaling_row(node_feature)
    norm_node_cfeature=rescaling_row(node_cfeature)
    
    norm_pair_feature0=pair_feature0/max_values0
    norm_pair_feature1=pair_feature1/max_values1
    norm_pair_feature2=pair_feature2/max_values2
    norm_pair_cfeature0=pair_cfeature0/cmax_values0
    norm_pair_cfeature1=pair_cfeature1/cmax_values1
    norm_pair_cfeature2=pair_cfeature2/cmax_values2    

    start_time=time.time()
    num_features = 2*len(norm_node_feature)+2*len(norm_node_cfeature)+3*len(norm_pair_feature0[0])+3*len(norm_pair_cfeature0[0])
    store_features = np.zeros((len(vertex_list), num_features))
    
    #print(f"shape: {norm_node_feature.shape}; {norm_node_cfeature.shape}; {norm_pair_feature0.shape}; {norm_pair_cfeature0.shape}")
    #print(f"store_features: {store_features.shape}")
    for id_v, curr_v in enumerate(vertex_list):

        vals=[]
        v1, v2 = int(curr_v[0]), int(curr_v[1])
        
        for ii in range(len(norm_node_feature)):
            vals.append(norm_node_feature[ii][v1])
            vals.append(norm_node_feature[ii][v2])

        for ii in range(len(norm_node_cfeature)):
            vals.append(norm_node_cfeature[ii][v1])
            vals.append(norm_node_cfeature[ii][v2]) 
             
        for ii in range(len(norm_pair_feature0[0])):
            vals.append(norm_pair_feature0[:,ii][id_v])
            vals.append(norm_pair_feature1[:,ii][id_v])
            vals.append(norm_pair_feature2[:,ii][id_v])
 
        for ii in range(len(norm_pair_cfeature0[0])):
            vals.append(norm_pair_cfeature0[:,ii][id_v])
            vals.append(norm_pair_cfeature1[:,ii][id_v])
            vals.append(norm_pair_cfeature2[:,ii][id_v])
            
        store_features[id_v] = vals

        if id_v%10**5==0: #if ii%10**4==0:

            #print(f'    compute_all_properties_of_list progress: ({time.time()-start_time} sec), {id_v/10**6}M/{len(vertex_list)/10**6}M')
            with open(logs_file_name+"_logs.txt", "a") as myfile:
                myfile.write(f'\n    compute_all_properties_of_list progress: ({time.time()-start_time} sec), {id_v/10**6}M/{len(vertex_list)/10**6}M')
            start_time=time.time()

    #print('Finish store_features') 
    with open(logs_file_name+"_logs.txt", "a") as myfile:
        myfile.write('\nFinish store_features')
    
    return store_features  