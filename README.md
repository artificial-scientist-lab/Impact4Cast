# Forecasting high-impact research topics via machine learning on evolving knowledge graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Paper Availability](https://img.shields.io/badge/paper-available-green)

**Authors:** [**Xuemei Gu**](mailto:xuemei.gu@mpl.mpg.de), [**Mario Krenn**](https://mpl.mpg.de/research-at-mpl/independent-research-groups/krenn-research-group/)
\
**Preprint:** [arXiv:2402.08640](https://arxiv.org/abs/2402.08640)

**Which scientific concepts, that have never been investigated jointly, will lead to the most impactful research?**

<img src="miscellaneous/Impact4Cast.png" alt="workflow" width="700"/>

> [!NOTE]\
> Full Dynamic Knowledge Graph and Datasets can be downloaded via [zenodo.org](https://zenodo.org/records/10692137)  

## <a name="ff">Prepare an evolving, citation-augmented knowledge graph</a>
### <a name="ff">Creating a list of scientific concepts</a>

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
 
### <a name="ff">Creating dynamic knowlegde graph</a>

<pre>
<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_dynamic_edges">create_dynamic_edges</a>
├── _get_openalex_workdata.py: Get metadata from <a href="https://openalex.org/">OpenAlex</a>)
├── _get_openalex_workdata_parallel_run1.py: Get parts of the metadata from OpenAlex (run in many parts)
├── get_concept_pairs.py: Create edges of the knowledge graph (edges carry the time and citation information).
├── merge_concept_pairs.py: Combining edges files
└── process_edge_to_pandas_frame.py: Post-processing, store the full dynamic knowledge graph
</pre>
<img src="miscellaneous/KnowledgeGraph.png" alt="workflow" width="800"/>


### <a name="ff">Prepare other data</a>
<pre>
.
├── prepare_unconnected_pair_solution.ipynb: Find unconnected concept pairs (for training, testing and evaluating)
├── prepare_adjacency_pagerank.py: Prepare dynamic knowledge graph and compute properties
│
├──<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_dynamic_concepts">create_dynamic_concepts</a>
│  ├── get_concept_citation.py: Create dynamic concepts from the knowledge graph (concepts carry the time and citation information). 
│  ├── merge_concept_citation.py: Combining dynamic concepts files
│  └── process_concept_to_pandas_frame.py: Post-processing, store the full dynamic concepts
│  ├── merge_concept_pairs.py: Combining dynamic concepts
│  └── process_edge_to_pandas_frame.py: Post-processing, store the full dynamic concepts
│
└──<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/prepare_eval_data">prepare_eval_data</a>
   ├── prepare_eval_feature_data.py: Prepare features of knowledge graph (for evaluation dataset)
   └── prepare_eval_feature_data_condition.py: Prepare features of knowledge graph (for evaluation dataset, conditioned on existence in the future)
</pre>

## <a name="ff">🤖Forecasting with Neural Network </a>
<img src="miscellaneous/Fig2_NeuralNet.png" alt="workflow" width="800"/>
<pre>
.
├── train_model_2019_run.py: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022).
├── train_model_2019_condition.py: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022, conditioned on existence in the future)
├── train_model_2019_individual_feature.py: Training neural network from 2016 -> 2019 (evaluated form 2019 -> 2022) on individual features
└── train_model_2022_run.py: Training 2019 -> 2022 (for real future predictions of 2025)
</pre>
 

<details>
  <summary><b>Feature descriptions for an unconnected pair of concepts (u, v)</b></summary>

<table>
    <tr>
        <th>Feature Type</th>
        <th>Feature Index</th>
        <th>Feature Description</th>
    </tr>
    <tr>
        <td rowspan="6" style="border-bottom: none;">node feature</td>
        <td>0-5</td>
        <td>the number of neighbours for vertices $u$ and $v$ in years $y$, $y-1$, $y-2$<br> denoted as: $N_{u,y}$, $N_{v,y}$, $N_{u,y-1}$, $N_{v,y-1}$, $N_{u, y-2}$, $N_{v, y-2}$ </td>
    </tr>
    <!-- Rows 2 to 10 for the first 'node feature' -->
    <tr><td>6-7</td><td>the number of new neighbors since 1 years prior to $y$ for vertices $u$ and $v$<br>denoted as: $N_{u,y}^{\Delta}$, $N_{v,y}^{\Delta}$
</td></tr>
    <tr><td>8-9</td><td>the number of new neighbors since 2 years prior to $y$ for vertices $u$ and $v$ <br>denoted as: $N_{u,y}^{\Delta 2}$, $N_{v,y}^{\Delta 2}$</td></tr>
    <tr><td>10-11</td><td>the rank of the number of new neighbors since 1 years prior to $y$ for vertices $u$ and $v$ <br>denoted as: $rN_{u,y}^{\Delta}$, $rN_{v,y}^{\Delta}$</td></tr>
    <tr><td>12-13</td><td>the rank of the number of new neighbors since 2 years prior to $y$ for vertices $u$ and $v$ <br>denoted as: $rN_{u,y}^{\Delta 2}$, $rN_{v,y}^{\Delta 2}$</td></tr>
    <tr><td>14-19</td><td>the PageRank score for vertices $u$ and $v$ in years $y$, $y-1$, $y-2$<br>denoted as: $PR_{u,y}$, $PR_{v,y}$, $PR_{u,y-1}$, $PR_{v,y-1}$, $PR_{u, y-2}$, $PR_{v, y-2}$ </td></tr>
    <!-- Starting the next 10 rows for the second 'node feature' -->
    <tr>
        <td rowspan="10" style="border-bottom: none;">node feature</td>
        <td>Data 11</td>
        <td>Data 11</td>
    </tr>
    <tr><td>Data 12</td><td>Data 12</td></tr>
    <tr><td>Data 13</td><td>Data 13</td></tr>
    <tr><td>Data 14</td><td>Data 14</td></tr>
    <tr><td>Data 15</td><td>Data 15</td></tr>
    <tr><td>Data 16</td><td>Data 16</td></tr>
    <tr><td>Data 17</td><td>Data 17</td></tr>
    <tr><td>Data 18</td><td>Data 18</td></tr>
    <tr><td>Data 19</td><td>Data 19</td></tr>
    <tr><td>Data 20</td><td>Data 20</td></tr>
    <!-- Starting the next 10 rows for the second 'node feature' -->
    <tr>
        <td rowspan="10" style="border-bottom: none;">pair feature</td>
        <td>Data 11</td>
        <td>Data 11</td>
    </tr>
    <tr><td>Data 12</td><td>Data 12</td></tr>
    <tr><td>Data 13</td><td>Data 13</td></tr>
    <tr><td>Data 14</td><td>Data 14</td></tr>
    <tr><td>Data 15</td><td>Data 15</td></tr>
    <tr><td>Data 16</td><td>Data 16</td></tr>
    <tr><td>Data 17</td><td>Data 17</td></tr>
    <tr><td>Data 18</td><td>Data 18</td></tr>
    <tr><td>Data 19</td><td>Data 19</td></tr>
    <tr><td>Data 20</td><td>Data 20</td></tr>
     <!-- Starting the next 10 rows for the second 'node feature' -->
    <tr>
        <td rowspan="10" style="border-bottom: none;">pair citation feature</td>
        <td>Data 11</td>
        <td>Data 11</td>
    </tr>
    <tr><td>Data 12</td><td>Data 12</td></tr>
    <tr><td>Data 13</td><td>Data 13</td></tr>
    <tr><td>Data 14</td><td>Data 14</td></tr>
    <tr><td>Data 15</td><td>Data 15</td></tr>
    <tr><td>Data 16</td><td>Data 16</td></tr>
    <tr><td>Data 17</td><td>Data 17</td></tr>
    <tr><td>Data 18</td><td>Data 18</td></tr>
    <tr><td>Data 19</td><td>Data 19</td></tr>
    <tr><td>Data 20</td><td>Data 20</td></tr>
</table>

</details>
 
