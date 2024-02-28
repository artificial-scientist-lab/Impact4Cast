import boto3
from botocore import UNSIGNED
from botocore.config import Config
import gzip
import jsonlines
import json
import os



# Function to filter the JSON objects by the desired keys
def filter_json_objects(json_obj, journal_paper, journal_paper_with_abstract):
    desired_keys = ['type', 'title', 'abstract_inverted_index', 'cited_by_count', 'counts_by_year', 'publication_year', 'publication_date']
    # Check if all the desired keys are in the JSON object
    if all(key in json_obj for key in desired_keys):
        if json_obj['type'] == 'journal-article' and json_obj['title'] not in [{}, None] and json_obj['publication_year'] not in [{}, None] and json_obj['publication_date'] not in [{}, None]:
            journal_paper += 1
            if json_obj['abstract_inverted_index'] not in [{}, None]:
                journal_paper_with_abstract += 1
                return {key: json_obj[key] for key in desired_keys}, journal_paper, journal_paper_with_abstract
    return None, journal_paper, journal_paper_with_abstract

# check whether a logs folder exists
logs_path = 'logs'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
   

journal_paper = 0
journal_paper_with_abstract = 0
# Create a local directory for the filtered files
local_base_folder = 'openalex_workdata_filtered'
os.makedirs(local_base_folder, exist_ok=True)
# Configure the S3 client for anonymous access
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Iterate through the objects in the specified S3 bucket and prefix
paginator = s3.get_paginator('list_objects_v2')

# Specify the S3 bucket and prefix (folder) as an example here
# change the folder files such that one can do parallel computing with many run code files
process_folder=['updated_date=2023-03-27','updated_date=2023-03-28'] # just an example

bucket_name = 'openalex'
prefix ='data/works/'

log_folders = os.path.join(logs_path, process_folder[0]+'_'+process_folder[-1].split('=')[1]+'_log.txt') 
for id_folder, folder in enumerate(process_folder):

    prefix ='data/works/'
    prefix = prefix+folder+'/'
    print(f"Process {prefix}, step%: {id_folder/len(process_folder)} \n")

    with open(log_folders, 'a') as log_file:
        log_file.write(f"Process {prefix}, progress: {id_folder/len(process_folder)} \n")

    for id_page, page in enumerate(paginator.paginate(Bucket=bucket_name, Prefix=prefix)):

        for id_obj, obj in enumerate(page['Contents']):

            if obj['Key'].split('/')[-1] == 'manifest':
                continue  # Skip the manifest file

            log_filename = os.path.join(logs_path, obj['Key'].split('/')[-2]+'_'+obj['Key'].split('/')[-1]+'_log.txt') 
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Page {id_page}, object {id_obj}; obj['Key']: {obj['Key']}\n")

            # Download and process the gzip-compressed JSON Lines file
            s3_object = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            
            with gzip.GzipFile(fileobj=s3_object['Body'], mode='r') as gz_file:
                with jsonlines.Reader(gz_file) as reader:
                    filtered_objects = []
                    for id_json, json_obj in enumerate(reader):
                        filtered_obj, journal_paper, journal_paper_with_abstract = filter_json_objects(json_obj, journal_paper, journal_paper_with_abstract)

                        if filtered_obj is not None:
                            filtered_objects.append(filtered_obj)

                        if id_json % 5000==0:
                            with open(log_filename, 'a') as log_file:
                                log_file.write(f"\n Processed {id_json} objects")

            # Prepare the local folder structure
            local_path_parts = obj['Key'].split('/')
            local_filtered_folder = os.path.join(local_base_folder, *local_path_parts[:-1])
            os.makedirs(local_filtered_folder, exist_ok=True)

            # Store the filtered objects in a new gzip-compressed JSON Lines file on the local computer
            filtered_file_name = f"filtered_{local_path_parts[-1]}"
            filtered_file_path = os.path.join(local_filtered_folder, filtered_file_name)
            with gzip.open(filtered_file_path, 'wt') as f:
                for item in filtered_objects:
                    f.write(json.dumps(item) + '\n')
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Finish writing {filtered_file_path}; until now, journal_paper: {journal_paper}; journal_paper_with_abstract: {journal_paper_with_abstract}\n")
            #print(f"Finish writing {obj['Key']}: {filtered_file_path} \n")

    with open(log_folders, 'a') as log_file:
        log_file.write(f"Finish {prefix}, progress: {id_folder/len(process_folder)} \nuntil now, journal_paper: {journal_paper}; journal_paper_with_abstract: {journal_paper_with_abstract}\n")

print(f"Finish writing all \n")