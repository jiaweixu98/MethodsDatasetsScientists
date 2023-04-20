# nohup python -u kg2rawdata.py > kg2rawdata.log 2>&1 &
import operator
from tqdm import tqdm
import pickle as pk

f = open("../data/paper_author.csv", "r")
paper_author = {}
author_paper = {}
for line in f:
    line = line.strip().split(',')
    if line[0] == 'PMID':
        continue
    elif line[0] in paper_author:
        paper_author[line[0]].append(line[3])
    else:
        paper_author[line[0]] = [line[3]]
    if line[3] in author_paper:
        author_paper[line[3]].append(line[0])
    else:
        author_paper[line[3]] = [line[0]]
f.close()

f = open("../data/paper_bioentity.csv", "r")
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

f = open("../data/article_datasets_breast_cancer_20230331.csv", "r")
paper_dataset = {}
dataset_paper = {}
for line in f:
    line = line.strip().split(',')
    if line[0] == 'PMID':
        continue
    paper_dataset[line[0]] = line[1].split('; ')
    for dataset in line[1].split('; '):
        if dataset in dataset_paper:
            dataset_paper[dataset].append(line[0])
        else:
            dataset_paper[dataset] = [line[0]]
f.close()

paper_citation = {}
s2id_citing_cited = pk.load(open('s2id_citing_cited.pkl', 'rb'))
pid_s2id = pk.load(open('pid_s2id.pkl', 'rb'))
s2id_pid = {}
for k, v in pid_s2id.items():
    s2id_pid[str(v)] = k
print('s2p finished')
for k, v in s2id_citing_cited.items():
    # k sid; V sid；下面转化为数字符号
    paper_citation[s2id_pid[k]] = list(map(lambda x: s2id_pid[x], v))

paper_ebd = {}
# s2id_citing_cited = pk.load(open('s2id_citing_cited.pkl','rb'))
s2id_embeddings = pk.load(open('pca_embeddings.pkl', 'rb'))
# s2id_embeddings = pk.load(open('pca_embeddings.pkl','rb'))

# for k, v in paper_bioentity.items():
#     print('paper_bioentity:', type(k), type(v))
#     break

# for k, v in s2id_embeddings.items():
#     print('s2id_embeddings:', type(k), type(v))
#     break

# for k, v in s2id_pid.items():
#     print('s2id_pid:', type(k), type(v))
#     break

for k, v in s2id_embeddings.items():
    # k sid; V sid；下面转化为数字符号
    paper_ebd[s2id_pid[str(k)]] = v
# citation 不用管，因为正常的里面，citation也不是每个都有的
# 只要保证，citation在这个set里就行
PID_set = paper_author.keys() & paper_bioentity.keys() & paper_ebd.keys() & paper_dataset.keys()
# print('len(PID_set)，是不是“300708？', len(PID_set))


# 重复上边的代码，重新弄一下；限定到PID
f = open("../data/paper_author.csv", "r")
paper_author = {}
author_paper = {}
for line in tqdm(f):
    line = line.strip().split(',')
    if line[0] not in PID_set:
        continue
    if line[0] in paper_author:
        paper_author[line[0]].append(line[3])
    else:
        paper_author[line[0]] = [line[3]]
    if line[3] in author_paper:
        author_paper[line[3]].append(line[0])
    else:
        author_paper[line[3]] = [line[0]]
f.close()
print('len(author_paper): ', len(author_paper))
print('len(paper_author): ', len(paper_author))

f = open("../data/paper_bioentity.csv", "r")
paper_bioentity = {}
bioentity_paper = {}
for line in f:
    line = line.strip().split(',')
    if line[0] not in PID_set:
        continue
    if line[0] in paper_bioentity:
        paper_bioentity[line[0]].append(line[1])
    else:
        paper_bioentity[line[0]] = [line[1]]
    if line[1] in bioentity_paper:
        bioentity_paper[line[1]].append(line[0])
    else:
        bioentity_paper[line[1]] = [line[0]]
f.close()
print('len(bioentity_paper): ', len(bioentity_paper))
print('len(paper_bioentity): ', len(paper_bioentity))


paper_new_citation = {}
for k, v in paper_citation.items():
    if k in PID_set:
        for i in v:
            if i in PID_set:
                if k in paper_new_citation:
                    paper_new_citation[k].append(i)
                else:
                    paper_new_citation[k] = [i]
print('len(paper_new_citation)',len(paper_new_citation))


f = open("../data/article_datasets_breast_cancer_20230331.csv", "r")
paper_dataset = {}
dataset_paper = {}
for line in f:
    line = line.strip().split(',')
    if line[0] not in PID_set:
        continue
    paper_dataset[line[0]] = line[1].split('; ')
    for dataset in line[1].split('; '):
        if dataset in dataset_paper:
            dataset_paper[dataset].append(line[0])
        else:
            dataset_paper[dataset] = [line[0]]
f.close()
print('len(paper_dataset)',len(paper_dataset))
print('len(dataset_paper)',len(dataset_paper))
# 至此，全部更新完毕。(ebd一定是最小的子集，其实不需要更新)

# 将id变为顺序数字index（Pcount）
PID_trans_Pcount = {}
Pcount_trans_PID = {}

#paper从0开始
count = 0
for PID in PID_set:
    PID_trans_Pcount[PID] = count
    Pcount_trans_PID[count] = PID
    count += 1
pk.dump(PID_trans_Pcount, open('PID_trans_Pcount.pkl', 'wb'))
pk.dump(Pcount_trans_PID, open('Pcount_trans_PID.pkl', 'wb'))
# author 从0开始


count = 0
AID_trans_Acount = {}
Acount_trans_AID = {}

for AID in author_paper.keys():
    AID_trans_Acount[AID] = count
    Acount_trans_AID[count] = AID
    count += 1
pk.dump(AID_trans_Acount, open('AID_trans_Acount.pkl', 'wb'))
pk.dump(Acount_trans_AID, open('Acount_trans_AID.pkl', 'wb'))

# bioentity 从0开始
count = 0
BID_trans_Bcount = {}
Bcount_trans_BID = {}
for BID in bioentity_paper.keys():
    BID_trans_Bcount[BID] = count
    Bcount_trans_BID[count] = BID
    count += 1
pk.dump(BID_trans_Bcount, open('BID_trans_Bcount.pkl', 'wb'))
pk.dump(Bcount_trans_BID, open('Bcount_trans_BID.pkl', 'wb'))


# dataset 从0开始
count = 0
DID_trans_Dcount = {}
Dcount_trans_DID = {}
for DID in dataset_paper.keys():
    DID_trans_Dcount[DID] = count
    Dcount_trans_DID[count] = DID
    count += 1
pk.dump(DID_trans_Dcount, open('DID_trans_Dcount.pkl', 'wb'))
pk.dump(Dcount_trans_DID, open('Dcount_trans_DID.pkl', 'wb'))

paper_c2author_c = {}
author_c2paper_c = {}
paper_c2bio_c = {}
bio_c2paper_c = {}
paper_c2citation_c = {}
paper_c2ebd = {}
paper_c2dataset_c = {}
dataset_c2paper_c = {}
for k, v in paper_author.items():
    # k,PId; V AID；下面转化为数字符号；实际上只转换bioentity应该就可以，在此全部转换，以防万一
    paper_c2author_c[PID_trans_Pcount[k]] = list(
        map(lambda x: AID_trans_Acount[x], v))
# 以下基本相同
for k, v in author_paper.items():
    author_c2paper_c[AID_trans_Acount[k]] = list(
        map(lambda x: PID_trans_Pcount[x], v))

for k, v in paper_bioentity.items():
    paper_c2bio_c[PID_trans_Pcount[k]] = list(
        map(lambda x: BID_trans_Bcount[x], v))

for k, v in bioentity_paper.items():
    bio_c2paper_c[BID_trans_Bcount[k]] = list(
        map(lambda x: PID_trans_Pcount[x], v))


for k, v in paper_dataset.items():
    paper_c2dataset_c[PID_trans_Pcount[k]] = list(
        map(lambda x: DID_trans_Dcount[x], v))

for k, v in dataset_paper.items():
    dataset_c2paper_c[DID_trans_Dcount[k]] = list(
        map(lambda x: PID_trans_Pcount[x], v))

for k, v in paper_new_citation.items():
    paper_c2citation_c[PID_trans_Pcount[k]] = list(
        map(lambda x: PID_trans_Pcount[x], v))

for k, v in paper_ebd.items():
    paper_c2ebd[PID_trans_Pcount[k]] = v

p_a_list_train_f = open("../data/p_a_list_train.txt", "w")
sorted_paper_c2author_c = sorted(
    paper_c2author_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_paper_c2author_c).items():
    p_a_list_train_f.write(str(k) + ":")
    for tt in range(len(v)-1):
        p_a_list_train_f.write(str(v[tt]) + ",")
    p_a_list_train_f.write(str(v[-1]))
    p_a_list_train_f.write("\n")
p_a_list_train_f.close()

a_p_list_train_f = open("../data/a_p_list_train.txt", "w")
sorted_author_c2paper_c = sorted(
    author_c2paper_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_author_c2paper_c).items():
    a_p_list_train_f.write(str(k) + ":")
    for tt in range(len(v)-1):
        a_p_list_train_f.write(str(v[tt]) + ",")
    a_p_list_train_f.write(str(v[-1]))
    a_p_list_train_f.write("\n")
a_p_list_train_f.close()

v_p_list_train_f = open("../data/v_p_list_train.txt", "w")
sorted_bio_c2paper_c = sorted(
    bio_c2paper_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_bio_c2paper_c).items():
    v_p_list_train_f.write(str(k) + ":")
    for tt in range(len(v)-1):
        v_p_list_train_f.write(str(v[tt]) + ",")
    v_p_list_train_f.write(str(v[-1]))
    v_p_list_train_f.write("\n")
v_p_list_train_f.close()

p_v_f = open("../data/p_v.txt", "w")
sorted_paper_c2bio_c = sorted(
    paper_c2bio_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_paper_c2bio_c).items():
    p_v_f.write(str(k) + ":")
    for tt in range(len(v)-1):
        p_v_f.write(str(v[tt]) + ",")
    p_v_f.write(str(v[-1]))
    p_v_f.write("\n")
p_v_f.close()


d_p_list_train_f = open("../data/d_p_list_train.txt", "w")
sorted_dataset_c2paper_c = sorted(
    dataset_c2paper_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_dataset_c2paper_c).items():
    d_p_list_train_f.write(str(k) + ":")
    for tt in range(len(v)-1):
        d_p_list_train_f.write(str(v[tt]) + ",")
    d_p_list_train_f.write(str(v[-1]))
    d_p_list_train_f.write("\n")
d_p_list_train_f.close()

p_d_f = open("../data/p_d.txt", "w")
sorted_paper_c2dataset_c = sorted(
    paper_c2dataset_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_paper_c2dataset_c).items():
    p_d_f.write(str(k) + ":")
    for tt in range(len(v)-1):
        p_d_f.write(str(v[tt]) + ",")
    p_d_f.write(str(v[-1]))
    p_d_f.write("\n")
p_d_f.close()


p_p_citation_list = open("../data/p_p_citation_list.txt", "w")
sorted_paper_c2citation_c = sorted(
    paper_c2citation_c.items(), key=operator.itemgetter(0))
for k, v in dict(sorted_paper_c2citation_c).items():
    p_p_citation_list.write(str(k) + ":")
    for tt in range(len(v)-1):
        p_p_citation_list.write(str(v[tt]) + ",")
    p_p_citation_list.write(str(v[-1]))
    p_p_citation_list.write("\n")
p_p_citation_list.close()

p_abstract_embed = open("../data/p_abstract_embed.txt", "w")
sorted_paper_c2ebd = sorted(paper_c2ebd.items(), key=operator.itemgetter(0))
p_abstract_embed.write(str(len(sorted_paper_c2ebd))+' 128'+"\n")
for k, v in dict(sorted_paper_c2ebd).items():
    p_abstract_embed.write(str(k)+' ')
    for tt in range(len(v)-1):
        p_abstract_embed.write(str(v[tt]) + " ")
    p_abstract_embed.write(str(v[-1]))
    p_abstract_embed.write("\n")
p_abstract_embed.close()
