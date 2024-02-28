# Forecasting high-impact research topics via machine learning on evolving knowledge graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper Availability](https://img.shields.io/badge/paper-available-green)

**Authors:** [**Xuemei Gu**](mailto:xuemei.gu@mpl.mpg.de), [**Mario Krenn**](https://mpl.mpg.de/research-at-mpl/independent-research-groups/krenn-research-group/)
\
**Preprint:** [arXiv:2402.08640](https://arxiv.org/abs/2402.08640)


_**Which scientific concepts, that have never been investigated jointly, will lead to the most impactful research?**_

<img src="miscellaneous/Impact4Cast.png" alt="workflow" width="700"/>

Datasets can be downloaded via [zenodo.org](https://zenodo.org/records/10692137)  

## <a name="ff">Prepare an evolving, citation-augmented knowledge graph</a>
 
### <a name="ff">1. Creating a list of scientific concepts</a>
<pre>
<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_concepts">create_concept</a>
│ 
├── <a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_concepts/Concept_Corpus">Concept_Corpus</a>
│   ├── s0_get_preprint_metadata.ipynb: Get metadata from chemRxiv, medRxiv, bioRxiv (<a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">arXiv data from Kaggle</a>)
│   ├── s1_make_metadate_arxivstyle.ipynb: Preprocessing metadata from different sources
│   ├── s2_combine_all_preprint_metadate.ipynb: Combining metadata
│   ├── s3_get_concepts.ipynb: Use NLP techniques (for instance <a href="https://github.com/csurfer/rake-nltk">RAKE</a> ) to extract concepts
│   └── s4_improve_concept.ipynb: Further improvements of full concept list
│   
└── <a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_concepts/Domain_Concept">Domain_Concept</a>
    ├── s0_prepare_optics_quantum_data.ipynb: Get papers for specific domain (optics and quantum physics in our case).
    ├── s1_split_domain_papers.py: Prepare data for parallelization.
    ├── s2_get_domain_concepts.py: Get domain-specific vertices in full concept list.
    ├── s3_merge_concepts.py: Postprocessing domain-specific concepts
    ├── s4_improve_concepts.ipynb: Further improve concept lists
    └── full_domain_concepts.txt: Final list of 37,960 concepts (represent vertices of knowledge graph)
</pre>
 
### <a name="ff">2. Creating dynamic knowlegde graph</a>

<pre>
<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_dynamic_edges">create_dynamic_edges</a>
│
├── _get_openalex_workdata.py: Get metadata from <a href="https://openalex.org/">OpenAlex</a>)
├── _get_openalex_workdata_parallel_run1.py: Get parts of the metadata from OpenAlex (run in many parts)
├── get_concept_pairs.py: Create edges of the knowledge graph (edges carry the time and citation information).
├── merge_concept_pairs.py: Combining edges files
└── process_edge_to_pandas_frame.py: Post-processing, store the full dynamic knowledge graph
</pre>
<img src="miscellaneous/KnowledgeGraph.png" alt="workflow" width="800"/>


## <a name="ff">Prepare other data</a>
<pre>
.
├── prepare_unconnected_pair_solution.ipynb: Find unconnected concept pairs (for training, testing and evaluating)
├── prepare_adjacency_pagerank.py: Prepare dynamic knowledge graph and compute properties
│
├──<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_dynamic_concepts">create_dynamic_concepts</a>
│  │
│  ├── get_concept_citation.py: Create dynamic concepts from the knowledge graph (concepts carry the time and citation information). 
│  ├── merge_concept_citation.py: Combining edges files
│  └── process_concept_to_pandas_frame.py: Post-processing, store the full dynamic concepts
│  ├── merge_concept_pairs.py: Combining edges files
│  └── process_edge_to_pandas_frame.py: Post-processing, store the full dynamic knowledge graph
│
└──<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/prepare_eval_data">prepare_eval_data</a>
   │
   ├── prepare_eval_feature_data.py: Prepare features of knowledge graph (for evaluation dataset)
   └── prepare_eval_feature_data_condition.py: Prepare features of knowledge graph (for evaluation dataset, conditioned on existence in the future)
</pre>

## <a name="ff">Forecasting with Neural Network </a>
<img src="miscellaneous/Fig2_NeuralNet.png" alt="workflow" width="800"/>
<pre>
.
├── train_model_2019_run.py: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022).
├── train_model_2019_condition.py: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022, conditioned on existence in the future)
├── train_model_2019_individual_feature.py: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022) on individual features
└── train_model_2022_run.py: Training 2019 -> 2022 (for real future predictions of 2025)
</pre>

## <a name="ff">Search Cliques </a>
<pre>
<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/search_cliques">search_cliques</a>
├── get_max_feature_for_norm_run.py: Calculating and normalizing the features of the entire knowledge graph.
├── nn_prediation_cliques_run.py: Neural-Network-based forecast of citation range for entire knowledge graph.
├── search_cliques.py: Search for cliques of predicted high-impact concepts.
└── result_clique_T0_IR_010.txt: Output of search.
</pre>
 
    

