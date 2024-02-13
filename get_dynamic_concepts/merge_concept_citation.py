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
log_files='log_merge_concept_citation.txt'

# define vertex_list foler
vertex_list_folder = 'concept_citation'
if not os.path.exists(vertex_list_folder):
    os.makedirs(vertex_list_folder)

list_file_names = os.listdir(vertex_list_folder) # List all files in the directory
vertex_file_name_unsorted = [file for file in list_file_names if file.endswith('.gz')]
vertex_lists_files = sorted(vertex_file_name_unsorted) # Sort the file list

full_vertex_lists = os.path.join(vertex_list_folder,'all_concept_citation.gz')  # vertex


with open(os.path.join(log_folder, log_files), 'a') as f:
    f.write(f'\nStart: {datetime.now()}\n')
    
    
full_vertices=[]
empty_count=0
for id_file, curr_vertex_files in enumerate(vertex_lists_files):

    with gzip.open(os.path.join(vertex_list_folder, curr_vertex_files), 'rb') as f: # load the vertex list
        vertex_data_list = pickle.load(f)

    if vertex_data_list!=[]:  # skip empty files
        full_vertices.extend(vertex_data_list)
    else:
        empty_count+=1
        print(f'Empty file: {curr_vertex_files}')

    # write to log file
    with open(os.path.join(log_folder, log_files), 'a') as f:
        f.write(f'Finish file: {curr_vertex_files}; v: {len(full_vertices)}; Processed: {(id_file+1)/len(vertex_lists_files)}; empty Num: {empty_count}\n')

# store the vertices list in a gz file
with gzip.open(full_vertex_lists, 'wb') as f:
    pickle.dump(full_vertices, f)

with open(os.path.join(log_folder, log_files), 'a') as f:
    f.write(f'\nFinish: {datetime.now()}\n')



