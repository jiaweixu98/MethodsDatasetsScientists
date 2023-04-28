# Understanding Breast Cancer Scientists with Datasets & Methods

 We use <a href="https://dl.acm.org/doi/pdf/10.1145/3292500.3330961"
                    style="text-decoration:none;color:#FFA15A;"> <b>HetGNN (Heterogeneous Graph Neural Network)</b></a> to represent node embeddings in the Breast Cancer area with the <a
                    href="https://www.nature.com/articles/s41597-020-0543-2"
                    style="text-decoration:none;color:#FFA15A;"> <b>Pubmed Knowledge Graph</b></a> Dataset.

If you have any questions, please report them in the issue tracker.

The demo website: [Breast Cancer: Scientists with Methods and Datasets.](<https://jiaweixu98.github.io/MethodsDatasetsScientists/>)

- #authors: 524,117

- #papers: 260,983

- #bio entities: 499,316

- #methods: 245

- #datasets: 308
- #Bridge2AI Scientists: 27

(Note: most of the ~ 524K authors lack the direct connection with methods and datasets in our data, only 6,824 authors have direct connections with methods and datasets.)

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

Because most authors (or papers) do not have methods neighbor nodes or datasets neighbor nodes, it is impossible to get a large fixed number of nodes (especially for datasets and methods) with the original configuration of [HetGNN](<https://dl.acm.org/doi/pdf/10.1145/3292500.3330961>). Therefore, we reduce the fixed number and reduce the rate of going back to the original node. In this way, we keep most nodes, and they have enough types and amount of neighbor nodes.

A concern: so many nodes have 0 data and method, but we try a random walk to get it a method/dataset node. Does that make sense? Thanks to the attention mechanism, the unimportant node will be ignored.

**In the subset folder, We only keep the paper nodes having the author, datasets, and methods. It is superfast but only has ~6000 authors.**

**A problem: only 1143 papers and ~6000 authors have both methods and datasets.**