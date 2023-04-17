# Understanding Breast Cancer Scientists with Datasets & Methods

## Heterogeneous Graph Neutral Network Method

We have five kinds of nodes: **author, paper, bioentity, dataset, method**, and their relationships:

1. author-paper: **writing**
2. paper-paper: **citing**
3. paper-bioentity: **mentioning**
4. paper-dataset: **using** (not all papers have datasets)
5. paper-method: **using** (not all papers have [OBI](https://obi-ontology.org/) methods)

## Raw Data Descriptor

We have not provided the data.

1. **"breast_cancaer_papers_PKG23.csv"** all papers about breast cancer in PKG, including the following fields: PMID, PubYear, ArticleTitle, Abstract.
    - 25,3891 unique entries

2. **"OBI_Extraction.csv"** includes the following fields: PMID, Mention(PKG), ShowName(OBI), OBI_id, Type, and Confidence.
    - 64,047 entries
    - unique paper: 47,819
    - unique dataset: 313

3. **"Dateset_Extraction_20230411.csv"** includes the following fields: PMID, Mention(PKG), ShowName(NAR/NIH/EBI),DBid, Confidence.
    - 8,447 entries
    - unique paper: 7,451
    - unique dataset: 259

4. **"paper_bioentity.csv"**
    - 5,298,578 entries
    - unique bioentity: 505,901
    - unique paper: 290,455

5. **"paper_author.csv"**
    - 1,690,409 entries
    - unique author: 556,952
    - unique paper: 303,911

## Notes

In the original HetGNN paper, a paper has some direct author neighbors and a venue neighbor.
Here in our situation, the paper may not have methods or datasets. We matched all the possible data with S2(318884/321216), 99.2%.


citation: 225,670
如果有节点是孤立的，直接丢弃。