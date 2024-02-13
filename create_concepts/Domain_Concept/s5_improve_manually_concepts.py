import os
from datetime import datetime, date

## repeated initial_num=0,1,2,...., start from 0, the second run will be 1, etc.
# the new_file_name is the final filtered concepts  
initial_num=16
file_name='full_concepts_for_openalex_'+str(initial_num)+'.txt'
curr_file = os.path.join("full_concepts",file_name) 
new_concept_list=[]

with open(curr_file, 'r') as file:
    lines = file.readlines()
    
concept_count=0
for idx, cc in enumerate(lines):
    #if cc[0]!='-':
    if "-" not in cc:
        new_concept_list.append(cc)
        concept_count+=1

now_time = datetime.now()
formatted_time = now_time.strftime("%d-%m-%Y %H:%M:%S")
print(f"{formatted_time}, Concepts: {concept_count} ; Remove: {idx-concept_count+1} ")

new_num=initial_num+1
new_file_name='full_concepts_for_openalex_'+str(new_num)+'.txt'
### 
with open(new_file_name, 'w') as file:
    for item in new_concept_list:
        file.write(f"{item}")