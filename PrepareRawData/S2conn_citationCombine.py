# 找到breast cancer graph 中的local citation relationships, 合并
# nohup python -u S2conn_citationCombine.py > S2conn_citationCombine.log 2>&1 &

import pickle as pk

s2id_citing_cited0 = pk.load(open('../../data/processedData/s2id_citing_citedT0.pkl','rb'))
s2id_citing_cited1 = pk.load(open('../../data/processedData/s2id_citing_citedT1.pkl','rb'))
s2id_citing_cited2 = pk.load(open('../../data/processedData/s2id_citing_citedT2.pkl','rb'))
s2id_citing_cited = {}

for k,v in s2id_citing_cited0.items():
    if k not in s2id_citing_cited:
        s2id_citing_cited[k] = v
    else:
        s2id_citing_cited[k] += v

for k,v in s2id_citing_cited1.items():
    if k not in s2id_citing_cited:
        s2id_citing_cited[k] = v
    else:
        s2id_citing_cited[k] += v

for k,v in s2id_citing_cited2.items():
    if k not in s2id_citing_cited:
        s2id_citing_cited[k] = v
    else:
        s2id_citing_cited[k] += v
print('len(s2id_citing_cited)',len(s2id_citing_cited))
pk.dump(s2id_citing_cited,open('../../data/processedData/s2id_citing_cited.pkl','wb'))