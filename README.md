# Impact4Cast

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2402.08640-b31b1b.svg)](https://arxiv.org/abs/2402.08640)
[![ICML AI4Science](https://img.shields.io/badge/ICML-2024-blue.svg)](https://openreview.net/forum?id=M1nqSqflLT&referrer=%5Bthe%20profile%20of%20Xuemei%20Gu%5D(%2Fprofile%3Fid%3D~Xuemei_Gu1))

**Which scientific concepts, that have never been investigated jointly, will lead to the most impactful research?**

📖 <u> Read our paper here: </u>\
[**Forecasting high-impact research topics via machine learning on evolving knowledge graphs**](https://arxiv.org/abs/2402.08640)\
*[Xuemei Gu](mailto:xuemei.gu@mpl.mpg.de), [Mario Krenn](mailto:mario.krenn@mpl.mpg.de)*

<img src="miscellaneous/Impact4Cast.png" alt="workflow" width="700"/>

> [!NOTE]\
> Full Dynamic Knowledge Graph and Datasets can be downloaded at [10.5281/zenodo.10692137](https://doi.org/10.5281/zenodo.10692137)  
> Dataset for Benchmark can be downloaded at [10.5281/zenodo.14527306](https://doi.org/10.5281/zenodo.14527306)

## <a name="ff">Prepare an evolving, citation-augmented knowledge graph</a>
### <a name="ff">Creating a list of scientific concepts</a>

<pre>
<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_concepts">create_concept</a>
│ 
├── <a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_concepts/Concept_Corpus">Concept_Corpus</a>
│   ├── s0_get_preprint_metadata.ipynb: Get metadata from chemRxiv, medRxiv, bioRxiv (<a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">arXiv data from Kaggle</a>)
│   ├── s1_make_metadate_arxivstyle.ipynb: Preprocessing metadata from different sources
│   ├── s2_combine_all_preprint_metadate.ipynb: Combining metadata
│   ├── s3_get_concepts.ipynb: Use NLP techniques (for instance <a href="https://github.com/csurfer/rake-nltk">RAKE</a>) to extract concepts
│   └── s4_improve_concept.ipynb: Further improvements of full concept list
│   
└── <a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/create_concepts/Domain_Concept">Domain_Concept</a>
    ├── s0_prepare_optics_quantum_data.ipynb: Get papers for specific domain (optics and quantum physics in our case).
    ├── s1_split_domain_papers.py: Prepare data for parallelization.
    ├── s2_get_domain_concepts.py: Get domain-specific vertices in full concept list.
    ├── s3_merge_concepts.py: Postprocessing domain-specific concepts
    ├── s4_improve_concepts.ipynb: Further improve concept lists
    ├── s5_improve_manually_concepts.py: Manually inspect the concepts in the very end for grammar, non-conceptual phrases, verbs, ordinal numbers, conjunctions, adverbials and so on, to improve quality
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
├── prepare_node_pair_citation_data_years.ipynb: Prepare citation data for both individual concept nodes and concept pairs for specific years
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
        <td>Number of neighbors for each node ($u$ or $v$) until the year $y$, $y{-}1$, $y{-}2$<br> denoted as $N_{u,y}$, $N_{v,y}$, $N_{u,y-1}$, $N_{v,y-1}$, $N_{u,y-2}$, and $N_{v,y-2}$, ordered as indices 0–5 </td>
    </tr>
    <!-- Rows 2 to 10 for the first 'node feature' -->
    <tr><td>6-7</td><td>Number of new neighbors for each node ($u$ or $v$) between year $y{-}1$ and $y$<br>i.e., $N_{u,y}{-}N_{u,y-1}$ and $N_{v,y}{-}N_{v,y-1}$
</td></tr>
    <tr><td>8-9</td><td>Number of new neighbors for each node ($u$ or $v$) between year $y{-}2$ and $y$ <br>i.e., $N_{u,y}{-}N_{u,y-2}$ and $N_{v,y}{-}N_{v,y-2}$</td></tr>
    <tr><td>10-11</td><td>Rank of the number of new neighbors for each node ($u$ or $v$) between year $y{-}1$ and $y$ <br>i.e., rank($N_{u,y}{-}N_{u,y-1}$) and rank($N_{v,y}{-}N_{v,y-1}$)</td></tr>
    <tr><td>12-13</td><td>Rank of the number of new neighbors for each node ($u$ or $v$) between year $y{-}2$ and $y$ <br>i.e., rank($N_{u,y}{-}N_{u,y-1}$) and rank($N_{v,y}{-}N_{v,y-2}$)</td></tr>
    <tr><td>14-19</td><td>PageRank scores of each node ($u$ or $v$) until the year $y$, $y{-}1$, $y{-}2$ <br>denoted and ordered as $\mathrm{PR}_{u,y}$, $\mathrm{PR}_{v,y}$, $\mathrm{PR}_{u,y-1}$, $\mathrm{PR}_{v,y-1}$, $\mathrm{PR}_{u,y-2}$ and $\mathrm{PR}_{v,y-2}$</td></tr>
    <!-- Starting the next 10 rows for the second 'node feature' -->
    <tr>
        <td rowspan="15" style="border-bottom: none;">node citation feature</td>
        <td>20-25</td>
        <td>Yearly citation for each node ($u$ or $v$) in year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{C}_{u,y}$, $\mathrm{C}_{v,y}$, $\mathrm{C}_{u,y-1}$, $\mathrm{C}_{v,y-1}$, $\mathrm{C}_{u,y-2}$ and $\mathrm{C}_{v,y-2}$</td>
    </tr>
    <tr><td>26-31</td><td>Total citation for each node ($u$ or $v$) since the first publication to the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{Ct}_{u,y}$, $\mathrm{Ct}_{v,y}$, $\mathrm{Ct}_{u,y-1}$, $\mathrm{Ct}_{v,y-1}$, $\mathrm{Ct}_{u,y-2}$ and $\mathrm{Ct}_{v,y-2}$ </td></tr>
    <tr><td>32-37</td><td>Total citations for each node ($u$ or $v$) in the last 3 years ending in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{Ct}^{\Delta 3}_{u,y}$, $\mathrm{Ct}^{\Delta 3}_{v,y}$, $\mathrm{Ct}^{\Delta 3}_{u,y{-}1}$, $\mathrm{Ct}^{\Delta 3}_{v,y{-}1}$, $\mathrm{Ct}^{\Delta 3}_{u,y{-}2}$, and $\mathrm{Ct}^{\Delta 3}_{v,y{-}2}$</td></tr>
    <tr><td>38-43</td><td>Number of papers mentioning node $u$ from the first publication to the year $y$, $y{-}1$, and $y{-}2$, similar for node $v$<br> denoted and ordered as $\mathrm{Pn}_{u,y}$, $\mathrm{Pn}_{v,y}$, $\mathrm{Pn}_{u,y-1}$, $\mathrm{Pn}_{v,y-1}$, $\mathrm{Pn}_{u,y-2}$, and $\mathrm{Pn}_{v,y-2}$</td></tr>
    <tr><td>44-49</td><td>Average yearly citations for each node ($u$ or $v$) in the year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Cm}_{u,y}$, $\mathrm{Cm}_{v,y}$, $\mathrm{Cm}_{u,y-1}$, $\mathrm{Cm}_{v,y-1}$, $\mathrm{Cm}_{u,y-2}$ and $\mathrm{Cm}_{v,y-2}$<br>e.g., $\mathrm{Cm}_{u,y}=\mathrm{C}_{u,y}/\mathrm{Pn}_{u,y}$</td></tr>
    <tr><td>50-55</td><td>Average total citations for each node ($u$ or $v$) since the first publications to the years $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Ctm}_{u,y}$, $\mathrm{Ctm}_{v,y}$, $\mathrm{Ctm}_{u,y-1}$, $\mathrm{Ctm}_{v,y-1}$, $\mathrm{Ctm}_{u,y-2}$ and $\mathrm{Ctm}_{v,y-2}$; e.g., $\mathrm{Ctm}_{u,y}=\mathrm{Ct}_{u,y}/\mathrm{Pn}_{u,y}$</td></tr>
    <tr><td>56-61</td><td>Average total citations for each node ($u$ or $v$) in the last 3 years ending in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{Ctm}^{\Delta 3}_{u,y}$, $\mathrm{Ctm}^{\Delta 3}_{v,y}$, $\mathrm{Ctm}^{\Delta 3}_{u,y-1}$, $\mathrm{Ctm}^{\Delta 3}_{v,y-1}$, $\mathrm{Ctm}^{\Delta 3}_{u,y-2}$ and $\mathrm{Ctm}^{\Delta 3}_{v,y-2}$<br>e.g., $\mathrm{Ctm}^{\Delta 3}_{u,y}=\mathrm{Ct}^{\Delta 3}_{u,y}/\mathrm{Pn}_{u,y}$</td></tr>
    <tr><td>62-63</td><td>New citations for each node ($u$ or $v$) between years $y{-}1$ and $y$ <br>i.e., $\mathrm{Ct}_{u,y}{-}\mathrm{Ct}_{u,y-1}$ and $\mathrm{Ct}_{v,y}{-}\mathrm{Ct}_{v,y-1}$</td></tr>
    <tr><td>64-65</td><td>New citations for each node ($u$ or $v$) between years $y{-}2$ and $y$<br>i.e., $\mathrm{Ct}_{u,y}{-}\mathrm{Ct}_{u,y-2}$ and $\mathrm{Ct}_{v,y}{-}\mathrm{Ct}_{v,y-2}$</td></tr>
    <tr><td>66-67</td><td>Rank of the new citations for each node ($u$ or $v$) between years $y{-}1$ and $y$<br>i.e., rank($\mathrm{C}_{u,y}{-}\mathrm{C}_{u,y-1}$) and rank($\mathrm{C}_{v,y}{-}\mathrm{C}_{v,y-1}$) </td></tr>
    <tr><td>68-69</td><td>Rank of the new citations for each node ($u$ or $v$) between years $y{-}2$ and $y$<br>i.e., rank($\mathrm{C}_{u,y}{-}\mathrm{C}_{u,y-2}$) and rank($\mathrm{C}_{v,y}{-}\mathrm{C}_{v,y-2}$)</td></tr>
    <tr><td>70-71</td><td>Number of papers mentioning nodes $u$ between years $y{-}1$ and $y$, similar for node $v$ <br>i.e., $\mathrm{PR}_{u,y}-\mathrm{PR}_{u,y-1}$ and $\mathrm{PR}_{v,y}-\mathrm{PR}_{v,y-1}$</td></tr>
    <tr><td>72-73</td><td>Number of papers mentioning nodes $u$ between years $y{-}2$ and $y$, similar for node $v$ <br>i.e., $\mathrm{PR}_{u,y}-\mathrm{PR}_{u,y-2}$ and $\mathrm{PR}_{v,y}-\mathrm{PR}_{v,y-2}$ </td></tr>
    <tr><td>74-75</td><td>Rank of the number of papers mentioning nodes $u$ between years $y{-}1$ and $y$, similar for node $v$<br>i.e., rank($\mathrm{PR}_{u,y}-\mathrm{PR}_{u,y-1}$) and rank($\mathrm{PR}_{v,y}-\mathrm{PR}_{v,y-1}$)</td></tr>
    <tr><td>76-77</td><td>Number of papers mentioning nodes $u$ between years $y{-}2$ and $y$, similar for node $v$ <br>i.e., rank($\mathrm{PR}_{u,y}-\mathrm{PR}_{u,y-2}$) and rank($\mathrm{PR}_{v,y}-\mathrm{PR}_{v,y-2}$)</td></tr>
    <!-- Starting the next 10 rows for the second 'node feature' -->
    <tr>
        <td rowspan="7" style="border-bottom: none;">pair feature</td>
        <td>78-80</td>
        <td>Number of shared neighbors between nodes $u$ and $v$ until the year $y$, $y{-}1$, $y{-}2$ <br> denoted and ordered as $\mathrm{Ns}_{y}$, $\mathrm{Ns}_{y-1}$ and $\mathrm{Ns}_{y-2}$; e.g., $\mathrm{Ns}_{y}=\mathrm{N}_{u,y} \cap \mathrm{N}_{v,y}$</td>
    </tr>
    <tr><td>81-83</td><td>Geometric similarity coefficient for the pair $(u, v)$ for the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{Geo}_{y}$, $\mathrm{Geo}_{y-1}$, and $\mathrm{Geo}_{y-2}$; e.g., $\mathrm{Geo}_{y} = \mathrm{Ns}_{y}^{2}/(\mathrm{N}_{u,y}\times \mathrm{N}_{v,y})$</td></tr>
    <tr><td>84-86</td><td>Cosine similarity coefficient for the pair $(u, v)$ for the year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Cos}_{y}$, $\mathrm{Cos}_{y-1}$, and $\mathrm{Cos}_{y-2}$; e.g.,  $\mathrm{Cos}_{y} = \sqrt{\mathrm{Geo}_{y}}$</td></tr>
    <tr><td>87-89</td><td>Simpson coefficient for the pair $(u, v)$ for the year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Sim}_{y}$, $\mathrm{Sim}_{y-1}$, and $\mathrm{Sim}_{y-2}$; e.g., $\mathrm{Sim}_{y} = \mathrm{Ns}_{y}/\min(\mathrm{N}_{u,y}, \mathrm{N}_{v,y})$</td></tr>
    <tr><td>90-92</td><td>Preferential attachment coefficient for the pair $(u, v)$ for the year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Pre}_{y}$, $\mathrm{Pre}_{y-1}$, and $\mathrm{Pre}_{y-2}$; e.g.,  $\mathrm{Pre}_{y} =\mathrm{N}_{u,y}\times \mathrm{N}_{v,y}$</td></tr>
    <tr><td>93-95</td><td>Sørensen–Dice coefficient for the pair $(u, v)$ for the year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Sor}_{y}$, $\mathrm{Sor}_{y-1}$, and $\mathrm{Sor}_{y-2}$; e.g., $\mathrm{Sor}_{y} = 2\mathrm{Ns}_{y}/(\mathrm{N}_{u,y}+\mathrm{N}_{v,y}$)</td></tr>
    <tr><td>96-98</td><td>Jaccard coefficient for the pair $(u, v)$ for the year $y$, $y{-}1$, $y{-}2$<br>denoted and ordered as $\mathrm{Jac}_{y}$, $\mathrm{Jac}_{y-1}$, and $\mathrm{Jac}_{y-2}$; e.g., $\mathrm{Jac}_{y} = \mathrm{Ns}_{y}/(\mathrm{N}_{u,y}+\mathrm{N}_{v,y}-\mathrm{Ns}_{y})$</td></tr>
     <!-- Starting the next 10 rows for the second 'node feature' -->
    <tr>
        <td rowspan="14" style="border-bottom: none;">pair citation feature</td>
        <td>99-101</td>
        <td>Ratio of the sum of citations received by nodes $u$ and $v$ until the year $y$ to the total number of papers mentioning either concept, similar for years $y-1$, $y-2$<br>denoted and ordered as $\mathrm{r1}_{y}$, $\mathrm{r1}_{y-1}$, and $\mathrm{r1}_{y-2}$; e.g., $\mathrm{r1}_{y}=(\mathrm{Ct}_{u,y}$ + $\mathrm{Ct}_{v,y}) / (\mathrm{Pn}_{u,y}+\mathrm{Pn}_{v,y})$.</td>
    </tr>
    <tr><td>102-104</td><td> Ratio of the product of citations received by nodes $u$ and $v$ until the year $y$ to the total number of papers mentioning either concept, similar for years $y-1$, $y-2$<br>denoted and ordered as $\mathrm{r2}_{y}$, $\mathrm{r2}_{y-1}$, and $\mathrm{r2}_{y-2}$; e.g., $\mathrm{r2}_{y}=(\mathrm{Ct}_{u,y} \times \mathrm{Ct}_{v,y}) / (\mathrm{Pn}_{u,y}+\mathrm{Pn}_{v,y})$ </td></tr>
    <tr><td>105-107</td><td>Sum of average citations received by nodes $u$ and $v$ in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{s}_{y}$, $\mathrm{s}_{y-1}$, and $\mathrm{s}_{y-2}$; e.g., $\mathrm{s}_{y}=\mathrm{Cm}_{u,y}+\mathrm{Cm}_{v,y}$</td></tr>
    <tr><td>108-110</td><td>Sum of average total citations received by nodes $u$ and $v$ from the first publication to the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{st}_{y}$, $\mathrm{st}_{y-1}$, and $\mathrm{st}_{y-2}$; e.g., $\mathrm{st}_{y}=\mathrm{Ctm}_{u,y}+\mathrm{Ctm}_{v,y}$</td></tr>
    <tr><td>111-113</td><td>Sum of the total citations received by nodes $u$ and $v$ in the last 3 years ending in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{st}^{\Delta 3}_{y}$, $\mathrm{st}^{\Delta 3}_{y-1}$, and $\mathrm{st}^{\Delta 3}_{y-2}$; e.g., $\mathrm{st}^{\Delta 3}_{y}=\mathrm{Ct}^{\Delta 3}_{u,y}+\mathrm{Ct}^{\Delta 3}_{v,y}$</td></tr>
    <tr><td>114-116</td><td>Sum of average total citations received by nodes $u$ and $v$ in the last 3 years ending in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{stm}^{\Delta 3}_{y}$, $\mathrm{stm}^{\Delta 3}_{y-1}$, and $\mathrm{stm}^{\Delta 3}_{y-2}$; e.g., $\mathrm{stm}^{\Delta 3}_{y}=\mathrm{Ctm}^{\Delta 3}_{u,y}+\mathrm{Ctm}^{\Delta 3}_{v,y}$</td></tr>
    <tr><td>117-119</td><td>Minimum number of citations received by either node $u$ or $v$ in years $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{minC}_{y}$, $\mathrm{minC}_{y-1}$, and $\mathrm{minC}_{y-2}$; e.g., $\mathrm{minC}_{y}=\min(\mathrm{C}_{u,y}, \mathrm{C}_{v,y})$</td></tr>
    <tr><td>120-122</td><td>Maximum number of citations received by either node $u$ or $v$ in years $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{maxC}_{y}$, $\mathrm{maxC}_{y-1}$, and $\mathrm{maxC}_{y-2}$; e.g., $\mathrm{maxC}_{y}=\max(\mathrm{C}_{u,y}, \mathrm{C}_{v,y})$</td></tr>
    <tr><td>123-125</td><td>Minimum number of total citations received by nodes $u$ and $v$ from the first publication to the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{minCt}_{y}$, $\mathrm{minCt}_{y-1}$, and $\mathrm{minCt}_{y-2}$; e.g., $\mathrm{minCt}_{y}= \min(\mathrm{Ct}_{u,y}, \mathrm{Ct}_{v,y})$ </td></tr>
    <tr><td>126-128</td><td>Maximum number of total citations received by nodes $u$ and $v$ from the first publication to the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{maxCt}_{y}$, $\mathrm{maxCt}_{y-1}$, and $\mathrm{maxCt}_{y-2}$; e.g., $\mathrm{maxCt}_{y}=\max(\mathrm{Ct}_{u,y}, \mathrm{Ct}_{v,y})$</td></tr>
    <tr><td>129-131</td><td>Minimum number of total citations received by nodes $u$ and $v$ in the last 3 years ending in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{minCt}^{\Delta 3}_{y}$, $\mathrm{minCt}^{\Delta 3}_{y-1}$, and $\mathrm{minCt}^{\Delta 3}_{y-2}$; e.g., $\mathrm{minCt}^{\Delta 3}_{y}= \min(\mathrm{Ct}^{\Delta 3}_{u,y}, \mathrm{Ct}^{\Delta 3}_{v,y})$.</td></tr>
    <tr><td>132-134</td><td>Maximum number of total citations received by nodes $u$ and $v$ in the last 3 years ending in the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{maxCt}^{\Delta 3}_{y}$, $\mathrm{maxCt}^{\Delta 3}_{y-1}$, and $\mathrm{maxCt}^{\Delta 3}_{y-2}$; e.g., $\mathrm{maxCt}^{\Delta 3}_{y}= \max(\mathrm{Ct}^{\Delta 3}_{u,y}, \mathrm{Ct}^{\Delta 3}_{v,y})$.</td></tr>
    <tr><td>135-137</td><td>Minimum number of papers mentioning the node $u$ or node $v$ from the first publication to the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{minPn}_{y}$, $\mathrm{minPn}_{y-1}$ and $\mathrm{minPn}_{y-2}$; e.g., $\mathrm{minPn}_{y}= \min(\mathrm{Pn}_{u,y}, \mathrm{Pn}_{v,y})$</td></tr> 
    <tr><td>138-140</td><td>Maximum number of papers mentioning the node $u$ or node $v$ from the first publication to the year $y$, $y{-}1$, and $y{-}2$<br>denoted and ordered as $\mathrm{maxPn}_{y}$, $\mathrm{maxPn}_{y-1}$ and $\mathrm{maxPn}_{y-2}$; e.g., $\mathrm{maxPn}_{y}= \max(\mathrm{Pn}_{u,y}, \mathrm{Pn}_{v,y})$</td></tr>  
</table>

</details>



### <a name="ff">Perform benchmarking</a>
One needs to download the data at [10.5281/zenodo.14527306](https://doi.org/10.5281/zenodo.14527306) and unzip the file in the benchmark_code folder.
<pre>
<a href="https://github.com/artificial-scientist-lab/Impact4Cast/tree/main/benchmark_code">benchmark_code</a>
├── loops_fcNN.py: fully connected neural network model
├── loops_transformer.py: transformer model
├── loops_tree.py: random forest model
├── loops_xgboost.py: XGBoost model
└── other python files: Post-processing, make the Figure 6-8 from the evaluation on different models.
</pre>

Three examples about 10M evaluation samples (2019-2022) with raw outputs from a neural network trained on 2016-2019 data (accessible at [10.5281/zenodo.14527306](https://doi.org/10.5281/zenodo.14527306)) are for producing Figure 11 in the fpr_example folder.


 
