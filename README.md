# Forecasting high-impact research topics via machine learning on evolving knowledge graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper Availability](https://img.shields.io/badge/paper-available-green)
![GitHub last commit](https://img.shields.io/github/last-commit/artificial-scientist-lab/Impact4Cast)

**Authors:** [**Xuemei Gu**](mailto:xuemei.gu@mpl.mpg.de), [**Mario Krenn**](https://mpl.mpg.de/research-at-mpl/independent-research-groups/krenn-research-group/)
\
**Preprint:** [arXiv:2402.08640](https://arxiv.org/abs/2402.08640)

<img src="miscellaneous/art_work.png" alt="workflow" width="550"/>


### Which scientific concepts, that have never been investigated jointly, will lead to the most impactful research?

Here we show how to predict the impact of onsets of ideas that have never been published by researchers. For that, we developed a large evolving knowledge graph built from more than 21 million scientific papers. It combines a semantic network created from the content of the papers and an impact network created from the historic citations of papers. Using machine learning, we can predict the dynamic of the evolving network into the future with high accuracy, and thereby the impact of new research directions. We envision that the ability to predict the impact of new ideas will be a crucial component of future artificial muses that can inspire new impactful and interesting scientific ideas.



## <a name="ff">Prepare an evolving, citation-augmented knowledge graph</a>
<img src="miscellaneous/KnowledgeGraph.png" alt="workflow" width="700"/>

### <a name="ff">1. Creating a list of scientific concepts</a>
**create_concept**: [/create_concept/](/create_concept/)
- **Concept_Corpus**:
  - `s0_get_preprint_metadata.ipynb`: Get paper metadata from chemRxiv, medRxiv, bioRxiv automatically (download [arXiv data from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) and store in same directory)
  - `s1_make_metadate_arxivstyle.ipynb`: Preprocessing metadata from different sources
  - `s2_combine_all_preprint_metadate.ipynb`: Combinging metadata
  - `s3_get_concepts.ipynb`: Use NLP techniques (for instance [RAKE](https://github.com/csurfer/rake-nltk)) to extract concepts
  - `s4_improve_concept.ipynb`: Further improvements of full concept list
    
- **Domain_Concept**:
  - `s0_prepare_optics_quantum_data.ipynb`: Get papers for specific domain (optics and quantum physics in our case).
  - `s1_split_domain_papers.py`: Prepare data for parallelization.
  - `s2_get_domain_concepts.py`: Get domain-specific vertices in full concept list.
  - `s3_merge_concepts.py`: Postprocessing domain-specific concepts.
  - `s4_improve_concepts.ipynb`: Further improve concept lists
  - `s5_improve_manually_concepts.py`: Further improve concept lists
  - `full_domain_concepts.txt`: Final list of 37,960 concepts (represent vertices of knowledge graph)
 
### <a name="ff">2. Creating dynamic knowlegde graph</a>
**create_dynamic_edges**: [/create_dynamic_edges/](/create_dynamic_edges/)
- `_get_openalex_workdata.py`: Get metadata from [OpenAlex](https://openalex.org/).
- `get_concept_pairs.py`: Create edges of knowledge graph (time and citation information).
- `merge_concept_pairs.py`: Post-processing
- `process_edge_to_pandas_frame.py`: Post-processing

 
## <a name="ff"> Utils Files and Prepare other data</a>
- `prepare_adjacency_pagerank.py`: Prepare dynamic knowledge graph and compute properties.
- `prepare_unconnected_pair_solution.ipynb`: Find unconnected concept pairs (for training, testing and evaluating)
  
**get_dynamic_concepts**: [/get_dynamic_concepts/](/get_dynamic_concepts/)
- `get_concept_citation.py`: Get citation data for nodes.
- `merge_concept_citation.py`: Post-processing
- `process_concept_to_pandas_frame.py`: Post-processing
  
**prepare_eval_data**: [/prepare_eval_data/](/prepare_eval_data/)
  - `prepare_eval_feature_data.py`: Prepare features of knowledge graph (for evaluation dataset)
  - `prepare_eval_feature_data_condition.py`: Prepare features of knowledge graph (for evaluation dataset, conditioned on existence in the future)


## <a name="ff">Forecasting with Neural Network </a>
<img src="miscellaneous/Fig2_NeuralNet.png" alt="workflow" width="800"/>

- `train_model_2019_run.py`: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022).
- `train_model_2019_condition.py`: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022, conditioned on existence in the future)
- `train_model_2022_run.py`: Training 2019 -> 2022 (for real future predictions of 2025)

## <a name="ff">Search Cliques </a>
**search_cliques**: [/search_cliques/](/search_cliques/)
 - `get_max_feature_for_norm_run.py`: Calculating and normalizing the features of the entire knowledge graph.
 - `nn_prediation_cliques_run.py`: Neural-Network-based forecast of citation range for entire knowledge graph.
 - `search_cliques.py`: Search for cliques of predicted high-impact concepts.
 - `result_clique_T0_IR_010.txt`: Output of search.
 
    

