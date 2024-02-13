# Impact4Cast: Forecasting High-Impact Research Topics
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper Availability](https://img.shields.io/badge/paper-available-green)
![Code Size](https://img.shields.io/github/languages/code-size/artificial-scientist-lab/Impact4Cast)
![Repo Size](https://img.shields.io/github/repo-size/artificial-scientist-lab/Impact4Cast)
 
<img src="miscellaneous/art_work.png" alt="workflow" width="500"/>


## <a name="ff">Creating a list of scientific concepts</a>
**create_concept**: [/create_concept/](/create_concept/)
- **Concept_Corpus**:
  - `s0_get_preprint_metadata.ipynb`
  - `s1_make_metadate_arxivstyle.ipynb`
  - `s2_combine_all_preprint_metadate.ipynb`
  - `s3_get_concepts.ipynb`
  - `s4_improve_concept.ipynb`
    
- **Domain_Concept**:
  - `s0_prepare_optics_quantum_data.ipynb`
  - `s1_split_domain_papers.py`
  - `s2_get_domain_concepts.py`
  - `s3_merge_concepts.py`
  - `s4_improve_concepts.ipynb`
  - `s5_improve_manually_concepts.py`
 
## <a name="ff">Creating dynamic knowlegde graph</a>
**create_dynamic_edges**: [/create_dynamic_edges/](/create_dynamic_edges/)
- `_get_openalex_workdata.py`
- `get_concept_pairs.py`
- `merge_concept_pairs.py`
- `process_edge_to_pandas_frame.py`
 
## <a name="ff">Prepare Other data used for Forecasting</a>
**get_dynamic_concepts**: [/get_dynamic_concepts/](/get_dynamic_concepts/)
- `get_concept_citation.py`
- `merge_concept_citation.py`
- `process_concept_to_pandas_frame.py`
  
**prepare_eval_data**: [/prepare_eval_data/](/prepare_eval_data/)
- `prepare_eval_feature_data.py`
- `prepare_eval_feature_data_condition.py`
  
**`prepare_adjacency_pagerank`** 


## <a name="ff">Forecasting the impact of new research connections</a>

 
    

