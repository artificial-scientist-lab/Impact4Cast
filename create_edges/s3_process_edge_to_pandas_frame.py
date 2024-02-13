import os
import sys
import pickle
import gzip
from datetime import datetime, date
import numpy as np
import pandas as pd
import time
import copy

log_folder = 'logs' # log folder
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    
data_folder="concept_pair"
data_file=os.path.join(data_folder,'all_concept_pairs.gz')   

store_folder="data_concept_graph"
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd) # get parent directory
new_dir_path = os.path.join(parent_dir, store_folder)
os.makedirs(new_dir_path, exist_ok=True)

store_data_file = os.path.join(new_dir_path, "full_dynamic_graph.parquet")



logsfile=os.path.join(log_folder,"logs_process_pairs.txt")
starting_time=time.time()
print(f'{datetime.now()}: read full graph')
with open(logsfile+'.txt', "a") as myfile:
    myfile.write(f'\n{datetime.now()}: read full graph') 

with gzip.open(data_file, 'rb') as f: # load the edge list
    full_dynamic_edges = pickle.load(f)

with open(logsfile+'.txt', "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, Total: {len(full_dynamic_edges)}; Elapsed time: {time.time() - starting_time} seconds\n")


# process the edge list to make each element with the same size
## [concept1, concept2, paper_time, total_citation, citation_per_year] 
## e.g., [7, 414, 10378, 1, [{'year': 2022, 'cited_by_count': 1}]] becomes [7, 414, 10378, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

starting_time = time.time()
full_dynamic_edges_copy = copy.deepcopy(full_dynamic_edges)
for i, item in enumerate(full_dynamic_edges):
    years_data = {year_data['year']: year_data['cited_by_count'] for year_data in item[4]}
    new_list = [years_data.get(year, 0) for year in range(2023, 2011, -1)] ## as cited_by_count only contains the last 10 years
    full_dynamic_edges_copy[i] = item[:4] + new_list

    if i % 200000 == 0:
        with open(logsfile+'.txt', "a") as myfile:
            myfile.write(f"\nProcessing item {i+1}/{len(full_dynamic_edges_copy)}")


time_start = time.time() 
full_graph=np.array(full_dynamic_edges_copy)
with open(logsfile+'.txt', "a") as myfile:
    myfile.write(f"\nDone, convert array; Elapsed time: {time.time() - time_start} seconds")
    
    
time_start = time.time()
full_graph_df = pd.DataFrame(full_graph, columns=['v1', 'v2', 'time', 'ct', 'c2023', 'c2022', 'c2021', 'c2020', 'c2019', 'c2018', 'c2017', 'c2016', 'c2015', 'c2014', 'c2013', 'c2012'])

full_graph_df.to_parquet(store_data_file, compression='gzip')

with open(logsfile+'.txt', "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, full_graph: {len(full_graph_df)}; Elapsed time: {time.time() - time_start} seconds")
    
 


