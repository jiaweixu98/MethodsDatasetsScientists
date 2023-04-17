# nohup python -u M_kg2rawdata.py > kg2rawdata.log 2>&1 &
import operator
from tqdm import tqdm
import pandas as pd
import pickle as pk
# read raw key value pairs data
paper_author = pk.load(open('../../data/processedData/paper_author.pkl', 'rb'))
author_paper = pk.load(open('../../data/processedData/author_paper.pkl', 'rb'))
paper_bioentity = pk.load(open('../../data/processedData/paper_bioentity.pkl', 'rb'))
bioentity_paper = pk.load(open('../../data/processedData/bioentity_paper.pkl', 'rb'))
paper_dataset = pk.load(open('../../data/processedData/paper_dataset.pkl', 'rb'))
dataset_paper = pk.load(open('../../data/processedData/dataset_paper.pkl', 'rb'))
paper_method = pk.load(open('../../data/processedData/paper_method.pkl', 'rb'))
method_paper = pk.load(open('../../data/processedData/method_paper.pkl', 'rb'))

