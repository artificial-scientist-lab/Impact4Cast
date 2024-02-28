# +
import pickle
import os
import time
from datetime import datetime, date
import random
 

# Create SemNet

if __name__ == '__main__':
    
 
    concept_folder="concept_seperate"
 
    
    total_file=78

    ## finish all 
    all_concepts_file = os.path.join(concept_folder,'all_concepts.pkl')  # edges
 
    #if not os.path.exists(all_concepts_file1):

    all_concepts=[]
 
    for id_file in range(total_file+1): # start from 0: 0-78
        
        file_ID = '{:02d}'.format(id_file)
        cur_concept_file=os.path.join(concept_folder, f'concept_{file_ID}.pkl')
 
        
        with open(cur_concept_file, 'rb') as file:  
            concept_info = pickle.load(file)

        all_concepts.extend(concept_info)
        
    with open(all_concepts_file, "wb") as output_file:
        pickle.dump(all_concepts, output_file)

 
