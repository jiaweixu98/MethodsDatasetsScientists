# nohup python -u result2pid.py > result2pid.log 2>&1 &
import pickle as pk
from tqdm import tqdm
Pcount_trans_PID = pk.load(open('../../../data/subsetHetGNNdata/Pcount_trans_PID.pkl', 'rb'))
Acount_trans_AID = pk.load(open('../../../data/subsetHetGNNdata/Acount_trans_AID.pkl', 'rb'))
Bcount_trans_BID = pk.load(open('../../../data/subsetHetGNNdata/Bcount_trans_BID.pkl', 'rb'))
Dcount_trans_DID = pk.load(open('../../../data/subsetHetGNNdata/Dcount_trans_DID.pkl', 'rb'))
Mcount_trans_MID = pk.load(open('../../../data/subsetHetGNNdata/Mcount_trans_MID.pkl', 'rb'))

f = open("../../../data/subsetHetGNNdata/98_node_embedding_datasetMethod.txt", "r")
fa = open("../../../data/subsetHetGNNdata/0422author_node_embedding.txt", "w")
fp = open("../../../data/subsetHetGNNdata/0422paper_node_embedding.txt", "w")
fb = open("../../../data/subsetHetGNNdata/0422entity_node_embedding.txt", "w")
fd = open("../../../data/subsetHetGNNdata/0422dataset_node_embedding.txt", "w")
fm = open("../../../data/subsetHetGNNdata/0422method_node_embedding.txt", "w")

for line in tqdm(f):
    line = line.strip().split(' ')
    if line[0][0] == 'a':
        fa.write(Acount_trans_AID[int(line[0][1:])])
        for i in line[1:]:
            fa.write(' '+i)
        fa.write('\n')
        continue
    if line[0][0] == 'b':
        fb.write(Bcount_trans_BID[int(line[0][1:])])
        for i in line[1:]:
            fb.write(' '+i)
        fb.write('\n')
        continue
    if line[0][0] == 'p':
        fp.write(Pcount_trans_PID[int(line[0][1:])])
        for i in line[1:]:
            fp.write(' '+i)
        fp.write('\n')
    if line[0][0] == 'd':
        fd.write(Dcount_trans_DID[int(line[0][1:])])
        for i in line[1:]:
            fd.write(' '+i)
        fd.write('\n')
    if line[0][0] == 'm':
        fm.write(Mcount_trans_MID[int(line[0][1:])])
        for i in line[1:]:
            fm.write(' '+i)
        fm.write('\n')
f.close()
fa.close()
fp.close()
fb.close()
fd.close()
fm.close()
