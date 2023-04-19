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
    - 253,891 unique entries

2. **"OBI_Extraction.csv"** includes the following fields: PMID, Mention(PKG), ShowName(OBI), OBI_id, Type, and Confidence.
    - 64,047 entries
    - unique paper: 47,819
    - unique method: 313

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

matched citation: 225,670
matched embeddings: 274,575


**Important Notes**

every paper should have its own text embedding, as well as the author.
paper with author & ebd: 260,983 (85.9% of paper with author). we can accept it.

It is important to consider a paper's author or its method, right?
If a paper does not have an author, just use the paper itself's embedding. Is that ok?
How to find the neighbors of a paper? Finding its author is the best solution. We must let the paper have at least one author.
Every paper has at least one author.
Other nodes (except paper) at least have a paper.

after constrined, the number of nodes:
len(paper_wauthor_bioentity) 253663
len(bioentity_wauthor_paper) 499316
len(paper_wauthor_dataset) 6544
len(dataset_wauthor_paper) 245
len(paper_wauthor_method) 45316
len(method_wauthor_paper) 308
len(pmid_citing_cited_constrained) 206621

if it is hard to get a fixed number of nodes (especially for datasets and methods), reduce the fixed number.