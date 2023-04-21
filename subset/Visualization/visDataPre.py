# nohup python -u visDataPre.py > visDataPre.log 2>&1 &
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.manifold import TSNE
import pickle as pk
# data read
author_emd = {}
dataset_emd = {}
method_emd = {}
fAuthor = open(
    "../../../data/subsetHetGNNdata/0422author_node_embedding.txt", "r")

print('start')

for line in tqdm(fAuthor):
    line = line.strip().split(' ')
    temp_ebd = list(map(float, line[1:]))
    author_emd[line[0]] = np.array(temp_ebd)
fAuthor.close()

fDataset = open(
    "../../../data/subsetHetGNNdata/0422dataset_node_embedding.txt", "r")
for line in tqdm(fDataset):
    line = line.strip().split(' ')
    temp_ebd = list(map(float, line[1:]))
    dataset_emd[line[0]] = np.array(temp_ebd)
fDataset.close()

fMethod = open(
    "../../../data/subsetHetGNNdata/0422method_node_embedding.txt", "r")
for line in tqdm(fMethod):
    line = line.strip().split(' ')
    temp_ebd = list(map(float, line[1:]))
    method_emd[line[0]] = np.array(temp_ebd)
fMethod.close()

# combine them
dataset_author_method_keys = list(dataset_emd.keys()) + list(author_emd.keys()) + list(method_emd.keys())
dataset_author_method_list = list(dataset_emd.values()) + list(author_emd.values()) + list(method_emd.values())
dataset_author_method_array = np.array(dataset_author_method_list)

# kmeans
dataset_author_method_Kmeans = KMeans(n_clusters=5, random_state=0).fit(dataset_author_method_array)
print('kmeans over')

clusters = dataset_author_method_Kmeans.predict(dataset_author_method_array)

dataset_author_method_kmeans_label = {}
for i in range(len(dataset_author_method_keys)):
    dataset_author_method_kmeans_label[dataset_author_method_keys[i]] = clusters[i]

dataset_cluster = {}
author_cluster = {}
method_cluster = {}
for i in dataset_emd.keys():
    dataset_cluster[i] = dataset_author_method_kmeans_label[i]
for i in author_emd.keys():
    author_cluster[i] = dataset_author_method_kmeans_label[i]
for i in method_emd.keys():
    method_cluster[i] = dataset_author_method_kmeans_label[i]
# tsne
dataset_author_method_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=100).fit_transform(dataset_author_method_array)
dataset_author_method_ebd = {}
for i in range(len(dataset_author_method_keys)):
    dataset_author_method_ebd[dataset_author_method_keys[i]] = dataset_author_method_embedded[i]

dataset_position = {}
author_position = {}
method_position = {}
for i in dataset_emd.keys():
    dataset_position[i] = dataset_author_method_ebd[i]
for i in author_emd.keys():
    author_position[i] = dataset_author_method_ebd[i]
for i in method_emd.keys():
    method_position[i] = dataset_author_method_ebd[i]

pk.dump(dataset_position, open('../../../data/subsetHetGNNdata/dataset_position.pkl', 'wb'))
pk.dump(author_position, open('../../../data/subsetHetGNNdata/author_position.pkl', 'wb'))
pk.dump(method_position, open('../../../data/subsetHetGNNdata/method_position.pkl', 'wb'))
# 上面处理好了数据集 和 作者的坐标

authorBreastSet = set(author_position.keys())
dataBreastSet = set(dataset_position.keys())
methodBreastSet = set(method_position.keys())

# 转换为dataframe
df_dataset = pd.DataFrame(dataset_position).T
df_author = pd.DataFrame(author_position).T
df_method = pd.DataFrame(method_position).T

df_author = df_author.rename({0: 'X', 1: 'Y'}, axis=1)
df_dataset = df_dataset.rename({0: 'X', 1: 'Y'}, axis=1)
df_method = df_method.rename({0: 'X', 1: 'Y'}, axis=1)

df_dataset.to_csv('../../../data/subsetHetGNNdata/dataset.csv', encoding='utf-8')
df_author.to_csv('../../../data/subsetHetGNNdata/author.csv', encoding='utf-8')
df_method.to_csv('../../../data/subsetHetGNNdata/method.csv', encoding='utf-8')

dataset = pd.read_csv('../../../data/subsetHetGNNdata/dataset.csv', index_col=0)
author = pd.read_csv('../../../data/subsetHetGNNdata/author.csv', index_col=0)
method = pd.read_csv('../../../data/subsetHetGNNdata/method.csv', index_col=0)

# add author name!
f = open("../../../data/subsetHetGNNdata/Authors.csv", "r")
author_name = {}
author_CareerAge = {}
author_PaperNum = {}
author_CitedNum = {}
author_mainAffiliation = {}
for line in tqdm(f):
    line = line.strip().split(',')
    if line[0] in authorBreastSet:
        author_name[int(line[0])] = line[4]
        author_CareerAge[int(line[0])] = line[1]
        author_PaperNum[int(line[0])] = line[2]
        author_CitedNum[int(line[0])] = line[3]
        author_mainAffiliation[int(line[0])] = str(line[5:]).strip(
            "\"[\]'").replace('\'', '').replace('\"', '').replace('  ', ' ')
f.close()

author['Name'] = pd.Series(author_name)
author['CareerAge'] = pd.Series(author_CareerAge)
author['PaperNum'] = pd.Series(author_PaperNum)
author['CitedNum'] = pd.Series(author_CitedNum)
author['MainAffi'] = pd.Series(author_mainAffiliation)

# add dataset name!
f = open("../../../data/Dateset_Extraction_20230411.csv", "r")
datasetIDName = {}
for line in tqdm(f):
    line = line.strip().split(',')
    if line[3] in dataBreastSet:
        datasetIDName[int(line[3])] = line[2]
f.close()
print('len(datasetIDName)',len(datasetIDName))
dataset['Name'] = pd.Series(datasetIDName)

# add obi name!
f = open("../../../data/OBI_Extraction.csv", "r")
obiIDName = {}
for line in tqdm(f):
    line = line.strip().split(',')
    if line[3] in methodBreastSet:
        obiIDName[int(line[3])] = line[2]
f.close()
print('len(obiIDName)',len(obiIDName))
method['Name'] = pd.Series(obiIDName)

dataset['clusterID'] = pd.Series(dataset_cluster)
author['clusterID'] = pd.Series(author_cluster)
method['clusterID'] = pd.Series(method_cluster)

# add author B2AI!
f = open("../../../data/subsetHetGNNdata/B2AiAuthors.csv", "r")
author_B2AI = {}
for i in authorBreastSet:
    author_B2AI[int(i)] = 0
for line in tqdm(f):
    line = line.strip().split(',')
    if line[0] in authorBreastSet and line[1] != '':
        author_B2AI[int(line[0])] = 1
f.close()
author['isB2AI'] = pd.Series(author_B2AI)

dataset.to_csv('../../../data/subsetHetGNNdata/dataset.csv', encoding='utf-8')
author.to_csv('../../../data/subsetHetGNNdata/author.csv', encoding='utf-8')
method.to_csv('../../../data/subsetHetGNNdata/method.csv', encoding='utf-8')