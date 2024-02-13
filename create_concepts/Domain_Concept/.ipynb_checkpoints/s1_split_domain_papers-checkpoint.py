import json
import time
from datetime import datetime, date
import pickle
import os
import math

log_folder='logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)


folder_name="data_seperate"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)   
    
data_folder="Concept_Corpus"
if not os.path.exists(data_folder):
    os.makedirs(data_folder) 

with open(os.path.join(data_folder,'arxiv_optics_quantum_paper_strings.pkl'), "rb") as f:
        get_all_paper_strings = pickle.load(f)
        
log_file = os.path.join(log_folder, 'split_papers_log.txt')
with open(log_file, 'a') as f:
    f.write(f"Seperate Optics and Quantum Papers: {len(get_all_paper_strings)}\n")


# Determine the number of parts needed
num_parts = math.ceil(len(get_all_paper_strings) / 1000)

# Store 1000 elements in each part file
for i in range(num_parts):
    time_starting=time.time()
    start_idx = i * 1000
    end_idx = min((i+1)*1000, len(get_all_paper_strings))
    part_data = get_all_paper_strings[start_idx : end_idx]
    part_file = os.path.join(folder_name, f'part_{i:02}.pkl')
    with open(part_file, 'wb') as f:
        pickle.dump(part_data, f)
    elapsed_time = time.time() - time_starting
    with open(log_file, 'a') as f:
        f.write(f"{i}: {(i+1)/num_parts}; Elapsed time: {elapsed_time} seconds \n")