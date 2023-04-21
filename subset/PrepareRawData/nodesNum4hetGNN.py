# generate the data for the input of HetGNN model.
# nohup python -u nodesNum4hetGNN.py > nodesNum4hetGNN.log 2>&1 &
import pickle as pk
def generate_data():
    # generate the data for the input of HetGNN model.
    print('generate the data for the input of HetGNN model.')
    # load k,v pairs.
    paper_author = pk.load(open('../../../data/processedData/paper_author.pkl', 'rb'))
    paper_bioentity = pk.load(open('../../../data/processedData/paper_bioentity.pkl', 'rb'))
    paper_dataset = pk.load(open('../../../data/processedData/paper_dataset.pkl', 'rb'))
    paper_method = pk.load(open('../../../data/processedData/paper_method.pkl', 'rb'))
    allPaperset = pk.load(open('../../../data/processedData/allPaperset.pkl', 'rb'))
    print('len(allPaperset))', len(allPaperset))
    # we let every paper has an author (at least)
    # paper must have embeddings!
    pmid_pca_embeddings = pk.load(open('../../../data/processedData/pmid_pca_embeddings.pkl', 'rb'))
    paper_constrain_list = []
    for paper in allPaperset:
        if (paper in paper_author) and (int(paper) in paper_dataset) and (int(paper) in paper_method) and (paper in pmid_pca_embeddings):
            paper_constrain_list.append(paper)
    paper_constrain_set = set(paper_constrain_list)
    print('len(paper_constrain_set)', len(paper_constrain_set))
    paper_wauthor_bioentity = {}
    for k,v in paper_bioentity.items():
        if k in paper_constrain_set:
            paper_wauthor_bioentity[k] = v
    bioentity_wauthor_paper = {}
    for k, v in paper_wauthor_bioentity.items():
        for bio in v:
            if bio in bioentity_wauthor_paper:
                bioentity_wauthor_paper[bio].append(k)
            else:
                bioentity_wauthor_paper[bio] = [k]
    paper_wauthor_dataset = {}
    for k, v in paper_dataset.items():
        k = str(k)
        if k in paper_constrain_set:
            paper_wauthor_dataset[k] = v
    dataset_wauthor_paper = {}
    for k, v in paper_wauthor_dataset.items():
        for dataset in v:
            if dataset in dataset_wauthor_paper:
                dataset_wauthor_paper[dataset].append(k)
            else:
                dataset_wauthor_paper[dataset] = [k]
    paper_wauthor_method = {}
    for k, v in paper_method.items():
        k = str(k)
        if k in paper_constrain_set:
            paper_wauthor_method[k] = v
    method_wauthor_paper = {}
    for k, v in paper_wauthor_method.items():
        for method in v:
            if method in method_wauthor_paper:
                method_wauthor_paper[method].append(k)
            else:
                method_wauthor_paper[method] = [k]
    # next, we see paper-ref
    pmid_citing_cited_constrained = {}
    pmid_citing_cited = pk.load(
        open('../../../data/processedData/pmid_citing_cited.pkl', 'rb'))
    for k,v in pmid_citing_cited.items():
        if k in paper_constrain_set:
            pmid_citing_cited_constrained[k] = []
            for ref in v:
                if ref in paper_constrain_set:
                    pmid_citing_cited_constrained[k].append(ref)
            if len(pmid_citing_cited_constrained[k]) == 0:
                pmid_citing_cited_constrained.pop(k)
    # next: embeddings:
    paper_constrain_ebd = {}
    for k, v in pmid_pca_embeddings.items():
        if k in paper_constrain_set:
            paper_constrain_ebd[k] = v    
    # next, authors
    paper_constrain_author = {}
    for k, v in paper_author.items():
        if k in paper_constrain_set:
            paper_constrain_author[k] = v
    author_constrain_paper = {}
    for k, v in paper_constrain_author.items():
        for author in v:
            if author in author_constrain_paper:
                author_constrain_paper[author].append(k)
            else:
                author_constrain_paper[author] = [k]

    print('len(paper_constrain_author)', len(paper_constrain_author))
    print('len(author_constrain_paper)', len(author_constrain_paper))
    print('len(paper_wauthor_bioentity)', len(paper_wauthor_bioentity))
    print('len(bioentity_wauthor_paper)', len(bioentity_wauthor_paper))
    print('len(paper_wauthor_dataset)', len(paper_wauthor_dataset))
    print('len(dataset_wauthor_paper)', len(dataset_wauthor_paper))
    print('len(paper_wauthor_method)', len(paper_wauthor_method))
    print('len(method_wauthor_paper)', len(method_wauthor_paper))
    print('len(pmid_citing_cited_constrained)', len(pmid_citing_cited_constrained))
    print('len(paper_constrain_ebd)', len(paper_constrain_ebd))

    pk.dump(paper_constrain_author, open(
        '../../../data/subsetHetGNNdata/paper_author.pkl', 'wb'))
    pk.dump(author_constrain_paper, open(
        '../../../data/subsetHetGNNdata/author_paper.pkl', 'wb'))
    pk.dump(paper_wauthor_bioentity, open('../../../data/subsetHetGNNdata/paper_bioentity.pkl', 'wb'))
    pk.dump(bioentity_wauthor_paper, open('../../../data/subsetHetGNNdata/bioentity_paper.pkl', 'wb'))
    pk.dump(paper_wauthor_dataset, open('../../../data/subsetHetGNNdata/paper_dataset.pkl', 'wb'))
    pk.dump(dataset_wauthor_paper, open('../../../data/subsetHetGNNdata/dataset_paper.pkl', 'wb'))
    pk.dump(paper_wauthor_method, open('../../../data/subsetHetGNNdata/paper_method.pkl', 'wb'))
    pk.dump(method_wauthor_paper, open('../../../data/subsetHetGNNdata/method_paper.pkl', 'wb'))
    pk.dump(pmid_citing_cited_constrained, open('../../../data/subsetHetGNNdata/pmid_citing_cited.pkl', 'wb'))
    pk.dump(paper_constrain_ebd, open('../../../data/subsetHetGNNdata/paper_constrain_ebd.pkl', 'wb'))
    




    return 0

if __name__ == '__main__':
    generate_data()