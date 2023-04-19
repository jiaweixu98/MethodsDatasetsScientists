# generate the data for the input of HetGNN model.
# nohup python -u kg2rawdata.py > kg2rawdata.log 2>&1 &
import pickle as pk
def generate_data():
    # generate the data for the input of HetGNN model.
    print('generate the data for the input of HetGNN model.')
    # load k,v pairs.
    paper_author = pk.load(open('../../data/processedData/paper_author.pkl', 'rb'))
    paper_bioentity = pk.load(open('../../data/processedData/paper_bioentity.pkl', 'rb'))
    paper_dataset = pk.load(open('../../data/processedData/paper_dataset.pkl', 'rb'))
    paper_method = pk.load(open('../../data/processedData/paper_method.pkl', 'rb'))
    allPaperset = pk.load(open('../../data/processedData/allPaperset.pkl', 'rb'))
    print('len(allPaperset))', len(allPaperset))
    # we let every paper has an author (at least)
    paper_list_has_author = []
    for paper in allPaperset:
        if paper in paper_author:
            paper_list_has_author.append(paper)
    paper_set_has_author = set(paper_list_has_author)
    # constrain author
    # paper_wauthor_bioentity
    # bioentity_wauthor_paper
    # paper_wauthor_dataset
    # dataset_wauthor_paper
    # paper_wauthor_method
    # method_wauthor_paper
    paper_wauthor_bioentity = {}
    for k,v in paper_bioentity.items():
        if k in paper_set_has_author:
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
        if k in paper_set_has_author:
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
        if k in paper_set_has_author:
            paper_wauthor_method[k] = v
    method_wauthor_paper = {}
    for k, v in paper_wauthor_method.items():
        for method in v:
            if method in method_wauthor_paper:
                method_wauthor_paper[method].append(k)
            else:
                method_wauthor_paper[method] = [k]
    # paper_wauthor_bioentity
    # bioentity_wauthor_paper
    # paper_wauthor_dataset
    # dataset_wauthor_paper
    # paper_wauthor_method
    # method_wauthor_paper
    # print('len(paper_wauthor_bioentity)', len(paper_wauthor_bioentity))
    # print('len(bioentity_wauthor_paper)', len(bioentity_wauthor_paper))
    # print('len(paper_wauthor_dataset)', len(paper_wauthor_dataset))
    # print('len(dataset_wauthor_paper)', len(dataset_wauthor_paper))
    # print('len(paper_wauthor_method)', len(paper_wauthor_method))
    # print('len(method_wauthor_paper)', len(method_wauthor_paper))

    # most of the data remained.
    




    return 0

if __name__ == '__main__':
    generate_data()