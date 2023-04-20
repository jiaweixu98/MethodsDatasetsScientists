# 第1号线程
# nohup python -u S2conn_citationT1.py > S2conn_citationT1.log 2>&1 &
# 找到breast cancer graph 中的local citation relationships
import jsonlines
from tqdm import tqdm
import pickle as pk

matched_s2id_set = pk.load(open('../../data/processedData/matched_s2id_set.pkl','rb'))

s2id_citing_cited = {}
for i in tqdm(range(10,20)):
    with jsonlines.open('/home/dell/kd_paper_data/data/SemanticScholar-20220913/full(v20220913)/citations/citationsP%d.jsonl'%i, mode='r') as reader:
        # item即为每一行的数据
        for item in reader:
            if item['citingcorpusid'] != None and item['citedcorpusid'] != None:
                if (item['citingcorpusid'] in matched_s2id_set) and (item['citedcorpusid'] in matched_s2id_set):
                    if item['citingcorpusid'] in s2id_citing_cited:
                        s2id_citing_cited[item['citingcorpusid']].append(item['citedcorpusid'])
                    else:
                        s2id_citing_cited[item['citingcorpusid']] = [item['citedcorpusid']]

print('len(s2id_citing_cited)',len(s2id_citing_cited))
pk.dump(s2id_citing_cited,open('../../data/processedData/s2id_citing_citedT1.pkl','wb'))