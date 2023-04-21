# Understanding Breast Cancer Scientists with Datasets & Methods

The demo website: [Breast Cancer: Scientists with Methods and Datasets.](<https://jiaweixu98.github.io/MethodsDatasetsScientists/>)

This is an early version of our project based on a subset of all authors. This version of the visualization demo is based on the 6,824 authors in the breast cancer area with datasets and methods.



- #authors: 6,824

- #papers: 1,143

- #bio entities: 8,633

- #methods: 89

- #datasets: 128

We will update the visualization of all the 524,117 authors in the breast cancer area days later because the large heterogeneous graph is still under training.

- #authors: 524,117

- #papers: 260,983

- #bio entities: 499,316

- #methods: 245

- #datasets: 308

(Note: most of the ~ 524K authors lack the direct connection with methods and datasets in our data, only 6,824 authors have direct connections with methods and datasets.)

Under Updating ...

## Heterogeneous Graph Neutral Network Method

We have five kinds of nodes: **author, paper, bioentity, dataset, method**, and their relationships:

1. author-paper: **writing**
2. paper-paper: **citing**
3. paper-bioentity: **mentioning**
4. paper-dataset: **using** (not all papers have datasets)
5. paper-method: **using** (not all papers have [OBI](https://obi-ontology.org/) methods)

## Raw Data Descriptor

We will provide the raw data upon request.

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

**A problem: only 1143 papers have both methods and datasets.**

## Notes

In the original HetGNN paper, a paper has some direct author neighbors and a venue neighbor.
Here in our situation, we let each paper have at least one author and the text embedding (from title, abstract, and citation relationships). The paper may not have methods or datasets.

Every paper should have its text embeddings, as well as the author.
Paper with author & embeddings: 260,983 (85.9% of paper with author).

- #papers with authors: 260,983
- #authors: 524,117
- #papers with bio entities 253,663
- #bio entities 499,316
- #papers with datasets 6,544
- #datasets 245
- #papers with methods 45,316
- #methods with papers 308
- #papers with references 205,761
- #papers with text embeddings 260,983

It is hard to get a fixed number of nodes (especially for datasets and methods), so we reduce the fixed number and reduce the rate of going back to the original node. We only keep the nodes with enough types and amount of neighbor nodes.

A concern: so many nodes have 0 data and method, but we try a random walk to get it a method/dataset node. Does that make sense? Thanks to the attention mechanism, the unimportant node will be ignored.

**We also employ an easy configuration: only keep the paper nodes having the author, datasets, and methods. Check the 
subset folder.**

**A problem: only 1143 papers have both methods and datasets.**

In subset

- paper 6,544
- paper_author 6,544
- author_constrain_paper 33,212
- paper_bioentity 6,518
- bioentity_paper 34,373
- paper_dataset 6,544
- dataset_paper 245
- paper_method 1,143
- method_paper 89
- pmid_citing_cited_constrained 3,286
- paper_ebd 6,544
