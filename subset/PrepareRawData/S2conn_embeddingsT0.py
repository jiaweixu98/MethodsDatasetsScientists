# 0 号进程
# nohup python -u S2conn_embeddingsT0.py > S2conn_embeddingsT0.log 2>&1 &
# 找到breast cancer graph 中的embd

import jsonlines
from tqdm import tqdm
import pickle as pk
matched_s2id_set = pk.load(open('../../data/processedData/matched_s2id_set.pkl','rb'))
matched_s2id_list = []
for i in matched_s2id_set:
    matched_s2id_list.append(int(i))
matched_s2id_list = set(matched_s2id_list)
print(len(matched_s2id_list))

s2id_embeddings = {}
for i in tqdm(range(15)):
    with jsonlines.open('/home/dell/kd_paper_data/data/SemanticScholar-20220913/full(v20220913)/embeddings/%d.jsonl'%i, mode='r') as reader:
        for item in reader:
            if item['corpusid'] in matched_s2id_list:
                s2id_embeddings[item['corpusid']] = eval(item['vector'])
print('len(s2id_embeddings)',len(s2id_embeddings))
pk.dump(s2id_embeddings,open('../../data/processedData/s2id_embeddings0.pkl','wb'))