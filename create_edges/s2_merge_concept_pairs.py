import glob
import gzip
import json
import os
import time
from datetime import datetime, date
import pickle
from functools import reduce
import random
 

log_folder = 'logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_files='log_merge_concept_pairs.txt'

# define edge_list foler
edge_list_folder = 'concept_pair'
if not os.path.exists(edge_list_folder):
    os.makedirs(edge_list_folder)

list_file_names = os.listdir(edge_list_folder) # List all files in the directory
edge_file_name_unsorted = [file for file in list_file_names if file.endswith('.gz')]
edge_lists_files = sorted(edge_file_name_unsorted) # Sort the file list

full_edge_lists = os.path.join(edge_list_folder,'all_concept_pairs.gz')  # edges


with open(os.path.join(log_folder, log_files), 'a') as f:
    f.write(f'\nStart: {datetime.now()}\n')
    
    
full_edges=[]
empty_count=0
for id_file, curr_edge_files in enumerate(edge_lists_files):

    with gzip.open(os.path.join(edge_list_folder, curr_edge_files), 'rb') as f: # load the edge list
        edge_data_list = pickle.load(f)

    if edge_data_list!=[]:  # skip empty files
        full_edges.extend(edge_data_list)
    else:
        empty_count+=1
        print(f'Empty file: {curr_edge_files}')

    # write to log file
    with open(os.path.join(log_folder, log_files), 'a') as f:
        f.write(f'Finish file: {curr_edge_files}; Edges: {len(full_edges)}; Processed: {(id_file+1)/len(edge_lists_files)}; empty Num: {empty_count}\n')

# store the edge list in a gz file
with gzip.open(full_edge_lists, 'wb') as f:
    pickle.dump(full_edges, f)

with open(os.path.join(log_folder, log_files), 'a') as f:
    f.write(f'\nFinish: {datetime.now()}\n')



