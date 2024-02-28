import json
import time
from datetime import datetime, date
import pickle
import os
import random

    
if __name__ == '__main__':

    log_folder='logs'
    try:
        os.mkdir(log_folder)
    except FileExistsError:
        pass 
    
    data_folder="Concept_Corpus"  
    data_seperate_folder="data_seperate"
    

    concept_folder="concept_seperate"
    try:
        os.mkdir(concept_folder)
    except FileExistsError:
        pass 


    concept_list_pkl = os.path.join(data_folder,'full_concept_list.pkl')

    with open(concept_list_pkl, 'rb') as file:
        all_concept_lists = pickle.load(file)
    

    random.seed() 
    total_file=78
    write_file=0
    cc=0
    while write_file <= total_file:

        curr_ID = random.randint(0, total_file)
        formatted_ID = '{:02d}'.format(curr_ID)
        data_file=os.path.join(data_seperate_folder, f'part_{formatted_ID}.pkl')
 
        concept_file=os.path.join(concept_folder, f'concept_{formatted_ID}.pkl')


        log_file = os.path.join('logs', 'log_'+formatted_ID+'.txt')
        if cc % 10 == 0:
            with open(log_file, 'a') as f:
                f.write(f'formatted_ID: {formatted_ID}; cc: {cc}, write_file num: {write_file}\n')
        cc+=1

        if not os.path.exists(concept_file):
             
            concepts_for_paper_list=[]

            with open(data_file, 'rb') as file:  # read paper 
                paper_info = pickle.load(file)
 
            concepts_at_least_one=[]
 
            # check all papers
            for id_paper, cur_paper in enumerate(paper_info):

                concepts_for_single_paper = []

                for id_concept, cur_concept in enumerate(all_concept_lists):

                    if cur_concept in cur_paper: # if the paper contains the concept
                        concepts_for_single_paper.append(cur_concept)  

                concepts_at_least_one.extend(concepts_for_single_paper) ## store the concepts from one paper
    

            finish_flag=0
            with open(concept_file, "wb") as output_file:
                pickle.dump(concepts_at_least_one, output_file)
                write_file+=1
                 

 