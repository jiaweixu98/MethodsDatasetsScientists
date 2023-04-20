# nohup python -u M_kg2rawdata.py > kg2rawdata.log 2>&1 &
import operator
from tqdm import tqdm
import pandas as pd
import pickle as pk

# read raw key value pairs data

breast_cancaer_papers_PKG23 = pd.read_csv('../../data/breast_cancaer_papers_PKG23.csv', index_col=0, dtype={'PMID': str})

PKG23_PMIDset = set(breast_cancaer_papers_PKG23['PMID'].unique().tolist())

paper_author = pk.load(open('../../data/processedData/paper_author.pkl', 'rb'))
# author_paper = pk.load(open('../../data/processedData/author_paper.pkl', 'rb'))
paper_bioentity = pk.load(open('../../data/processedData/paper_bioentity.pkl', 'rb'))
# bioentity_paper = pk.load(open('../../data/processedData/bioentity_paper.pkl', 'rb'))
paper_dataset = pk.load(open('../../data/processedData/paper_dataset.pkl', 'rb'))
# dataset_paper = pk.load(open('../../data/processedData/dataset_paper.pkl', 'rb'))
paper_method = pk.load(open('../../data/processedData/paper_method.pkl', 'rb'))
# method_paper = pk.load(open('../../data/processedData/method_paper.pkl', 'rb'))

# all index to string, see all the paper
combinedPaperset = set(list(map(str, paper_author.keys()))) | set(list(map(str, paper_bioentity.keys()))) | set(list(map(str, paper_dataset.keys()))) | set(list(map(str, paper_method.keys())))

# print(len(combinedPaperset))
print('union', len(PKG23_PMIDset | combinedPaperset))  # union 321216
# print('intersection', len(PKG23_PMIDset & combinedPaperset) ) #intersection 242407

allPaperset = PKG23_PMIDset | combinedPaperset
# just use all paper to match, take it easy.
pk.dump(allPaperset, open('../../data/processedData/allPaperset.pkl', 'wb'))