import glob
import gzip
import json
import os
import time
from datetime import datetime, date
import pickle
from functools import reduce
import random
import re


def get_single_article_string(article):
        
    curr_title=article['title']
    abstract_inverted_index = article['abstract_inverted_index']

    # Flatten the inverted index into a list of (position, word) tuples
    position_word_list = [(position, word) for word, positions in abstract_inverted_index.items() for position in positions]

    # Sort the list by position and extract the words
    sorted_abstract = sorted(position_word_list)
    curr_abstract = ' '.join(word for position, word in sorted_abstract)

    # Replace strings according to the replace_pairs list
    replace_pairs=[['\n',' '],['-',' '],[' \" a','oa'],['\" a','ae'],['\"a','ae'],[' \" o','oe'],['\" o','oe'],['\"o','oe'],[' \" u','ue'],['\" u','ue'],['\"u','ue'],[' \' a','a'],[' \' e','e'],[' \' o','o'],["\' ", ""],["\'", ""],['  ',' '],['  ',' ']]

    article_string=(curr_title +' '+ curr_abstract).lower()
    article_string = reduce(lambda text, pair: text.replace(pair[0], pair[1]), replace_pairs, article_string)

    return article_string

# Define a sorting key function to extract the date and part number from the path
def get_date_and_part_from_path(path):
    date_folder = os.path.dirname(path)
    date_str = date_folder.split('=')[-1]

    file_name = os.path.basename(path)
    part_str = file_name.split('_')[-1].split('.')[0]

    return date_str, int(part_str)

def extract_id(filename):
    match = re.search(r'log_concept_part_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


# define a log foler
log_folder = 'logs_concept'
# define edge_list foler
vertex_folder = 'concept_citation'
vertex_folder_log = 'concept_citation_log'

try:
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    if not os.path.exists(vertex_folder):
        os.makedirs(vertex_folder)
        
    if not os.path.exists(vertex_folder_log):
        os.makedirs(vertex_folder_log)

except FileExistsError:
    pass


data_folder="data_concept_graph"
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
concept_folder = os.path.join(parent_dir, data_folder)


#project_path="/u/xmgu/projects/semnet_openalex"
#base_folder=os.path.join(project_path,'openalex_workdata_filtered/data/works/')

# Define the base folder, date pattern and file pattern
base_folder = 'openalex_workdata_filtered/data/works/'
date_pattern = 'updated_date=*'
file_pattern = 'filtered_part_*.gz'

# Find all the files matching the pattern
file_paths = glob.glob(f'{base_folder}/{date_pattern}/{file_pattern}')
# Sort the file_paths list in ascending order based on the date and part number
file_paths = sorted(file_paths, key=get_date_and_part_from_path)

# Define the date range or specific folders to include
start_date = datetime.strptime("2022-12-20", "%Y-%m-%d")
end_date = datetime.strptime("2023-03-28", "%Y-%m-%d")

# Filter the file_paths list based on the date range or specific folders
curr_run_file_paths = [path for path in file_paths if start_date <= datetime.strptime(get_date_and_part_from_path(path)[0], "%Y-%m-%d") <= end_date]

rnd_time=random.random()*50
time.sleep(rnd_time)

# Read all concepts from full_final_concepts/full_domain_concept.txt
concepts_files = os.path.join(concept_folder, 'full_domain_concept.txt')
with open(concepts_files, 'r') as file:
    full_concepts = [concept.strip() for concept in file.readlines()]

# Define a list to store the vertex lists
paper_starting_date = date(1990,1,1) 

write_file=0

rnd_time=random.random()*60
time.sleep(rnd_time)

while write_file <=len(curr_run_file_paths):
    
    curr_ID = random.randint(0, len(curr_run_file_paths)-1) # get a random number between 0 and the number of files

    formatted_ID = '{:03d}'.format(curr_ID)

    edge_file=os.path.join(vertex_folder, 'concept_part_'+formatted_ID+'.gz')
    edge_file_log=os.path.join(vertex_folder_log, 'concept_part_'+formatted_ID+'.txt')
    log_file_txt=os.path.join(log_folder, 'log_concept_part_'+formatted_ID+'.txt')
    log_file_txt_finish=os.path.join(log_folder, 'log_concept_part_'+formatted_ID+'_finish.txt')
    log_file_txt_empty=os.path.join(log_folder, 'log_concept_part_'+formatted_ID+'_empty.txt')

    if not os.path.exists(log_file_txt):
        current_time=datetime.now()
        open(log_file_txt, 'a').close()
        
        file_path=curr_run_file_paths[curr_ID]
        with open(log_file_txt, 'a') as log_file:
            log_file.write(f'Current time: {current_time}; Number of files: {len(curr_run_file_paths)}; Number of concepts: {len(full_concepts)}\n\n')
            log_file.write(f'Start the File: {file_path}; Current time: {datetime.now()} \n\n')

        with gzip.open(file_path, 'rt') as file:
            lines = file.readlines()

            if not lines: # if lines is empty
                print(f'File {file_path} is empty')
                write_file+=1
                with open(log_file_txt_empty, 'a') as log_file:
                    log_file.write(f'Current File: {file_path}; Paper: {len(lines)}; File is Empty!\n')
                
            else:
                edge_lists=[]
                for id_line, line in enumerate(lines):
                    time_start_line=time.time()

                    article_object = json.loads(line) # Load the JSON object
                    get_date = datetime.strptime(article_object['publication_date'], "%Y-%m-%d").date()
                    curr_paper_time = (get_date - paper_starting_date).days
                    curr_all_citations=article_object['cited_by_count']
                    curr_citations_per_year=article_object['counts_by_year']
                    curr_article=get_single_article_string(article_object)


                    # Check if the article contains any of the concepts
                    concepts_for_single_paper=[]
                    for id_concept, concept in enumerate(full_concepts):
                        if concept in curr_article: # if the paper contains the concept; then store its concept index 
                            concepts_for_single_paper.append(id_concept)

                    for ii in range(len(concepts_for_single_paper)):
                        edge_lists.append([concepts_for_single_paper[ii],curr_paper_time,curr_all_citations,curr_citations_per_year])

                    if id_line % 10000 == 0:
                        with open(log_file_txt, 'a') as log_file:
                            log_file.write(f'Current File: {file_path}; Paper: {len(lines)}; Processed: {(id_line+1)/len(lines)}; time: {time.time()-time_start_line}\n')

                # Finish the current file, then store edge_lists to a pickle file
                with gzip.open(edge_file, 'wb') as output_file:
                    pickle.dump(edge_lists, output_file)
                    write_file+=1

                with open(edge_file_log, 'a') as log_file:
                    log_file.write(f'\nconcept_list={len(edge_lists)}')
                        
                with open(log_file_txt, 'a') as log_file:
                    log_file.write(f'\n\nFinish Time: {datetime.now()}; Current File: {file_path}; Processed: {write_file}/{len(curr_run_file_paths)}, i.e., {write_file/len(curr_run_file_paths)} \n')
                
                with open(log_file_txt_finish, 'a') as log_file:
                    log_file.write(f'\n\nFinish Time: {datetime.now()}; Current File: {file_path} \n')
                
                rnd_time=random.random()*5
                time.sleep(rnd_time)
                
    else:
        # Match file patterns
        finish_pattern = os.path.join(log_folder, 'log_concept_part_*_finish.txt')
        empty_pattern = os.path.join(log_folder, 'log_concept_part_*_empty.txt')
        finished_files = [f for f in glob.glob(finish_pattern) if extract_id(f) in range(0, len(curr_run_file_paths))]
        empty_files = [f for f in glob.glob(empty_pattern) if extract_id(f) in range(0, len(curr_run_file_paths))]

        # Count files that match each pattern
        total_files = len(finished_files) + len(empty_files)

        # Check if the total count is larger than 391
        if total_files >= len(curr_run_file_paths):
            print(f"{datetime.now()}:Finish run!")
            break

            

with open("job_finish.txt", 'a') as f:
    f.write(f'\nFinish all: {datetime.now()}\n')





