# nohup python -u M_kg2rawdata.py > kg2rawdata.log 2>&1 &
import operator
from tqdm import tqdm
import pandas as pd
import pickle as pk

# generate paper-author mapping
# f = open("../../data/paper_author.csv", "r")
# paper_author = {}
# author_paper = {}
# for line in f:
#     line = line.strip().split(',')
#     if line[0] == 'PMID':
#         continue
#     elif line[0] in paper_author:
#         paper_author[line[0]].append(line[3])
#     else:
#         paper_author[line[0]] = [line[3]]
#     if line[3] in author_paper:
#         author_paper[line[3]].append(line[0])
#     else:
#         author_paper[line[3]] = [line[0]]
# f.close()

# print(len(author_paper))
# print(len(paper_author))
# pk.dump(paper_author, open('../../data/processedData/paper_author.pkl', 'wb'))
# paper_author = pk.load(open('../../data/processedData/paper_author.pkl', 'rb'))
# print(len(paper_author))
# pk.dump(author_paper, open('../../data/processedData/author_paper.pkl', 'wb'))
# author_paper = pk.load(open('../../data/processedData/author_paper.pkl', 'rb'))
# print(len(author_paper))


# generate paper-bioentity mapping
f = open("../../data/paper_bioentity.csv", "r")
paper_bioentity = {}
bioentity_paper = {}
for line in f:
    line = line.strip().split(',')
    if line[0] == 'PMID':
        continue
    elif line[0] in paper_bioentity:
        paper_bioentity[line[0]].append(line[1])
    else:
        paper_bioentity[line[0]] = [line[1]]
    if line[1] in bioentity_paper:
        bioentity_paper[line[1]].append(line[0])
    else:
        bioentity_paper[line[1]] = [line[0]]
f.close()

print(len(paper_bioentity))
print(len(bioentity_paper))
pk.dump(paper_bioentity, open(
    '../../data/processedData/paper_bioentity.pkl', 'wb'))
paper_bioentity = pk.load(
    open('../../data/processedData/paper_bioentity.pkl', 'rb'))
print(len(paper_bioentity))
pk.dump(bioentity_paper, open('../../data/processedData/bioentity_paper.pkl', 'wb'))
bioentity_paper = pk.load(open('../../data/processedData/bioentity_paper.pkl', 'rb'))
print(len(bioentity_paper))

# generate paper-dataset mapping
Dateset_Extraction_20230411 = pd.read_csv(
    '../../data/Dateset_Extraction_20230411.csv', index_col=0)
paper_dataset = {}
dataset_paper = {}
for index, row in Dateset_Extraction_20230411.iterrows():
    if index in paper_dataset:
        paper_dataset[index].append(row['DBid'])
    else:
        paper_dataset[index] = [row['DBid']]
    if row['DBid'] in dataset_paper:
        dataset_paper[row['DBid']].append(index)
    else:
        dataset_paper[row['DBid']] = [index]

print(len(paper_dataset))
print(len(dataset_paper))
pk.dump(paper_dataset, open(
    '../../data/processedData/paper_dataset.pkl', 'wb'))
paper_dataset = pk.load(
    open('../../data/processedData/paper_dataset.pkl', 'rb'))
print(len(paper_dataset))
pk.dump(dataset_paper, open(
    '../../data/processedData/dataset_paper.pkl', 'wb'))
dataset_paper = pk.load(
    open('../../data/processedData/dataset_paper.pkl', 'rb'))
print(len(dataset_paper))

# generate paper-dataset mapping
OBI_Extraction = pd.read_csv(
    '../../data/OBI_Extraction.csv', index_col=0)
paper_method = {}
method_paper = {}
for index, row in OBI_Extraction.iterrows():
    if index in paper_method:
        paper_method[index].append(row['OBI_id'])
    else:
        paper_method[index] = [row['OBI_id']]
    if row['OBI_id'] in method_paper:
        method_paper[row['OBI_id']].append(index)
    else:
        method_paper[row['OBI_id']] = [index]

print(len(paper_method))
print(len(method_paper))

pk.dump(paper_method, open(
    '../../data/processedData/paper_method.pkl', 'wb'))
paper_method = pk.load(
    open('../../data/processedData/paper_method.pkl', 'rb'))
print(len(paper_method))
pk.dump(method_paper, open(
    '../../data/processedData/method_paper.pkl', 'wb'))
method_paper = pk.load(
    open('../../data/processedData/method_paper.pkl', 'rb'))
print(len(method_paper))
