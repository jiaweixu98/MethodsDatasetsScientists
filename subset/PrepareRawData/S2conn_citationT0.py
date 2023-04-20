# 第0号线程
# nohup python -u S2conn_citationT0.py > S2conn_citationT0.log 2>&1 &
# 找到breast cancer graph 中的local citation relationships
import jsonlines
from tqdm import tqdm
import pickle as pk

pid_s2id0 = pk.load(open('../../data/processedData/pid_s2idT0.pkl','rb'))
pid_s2id1 = pk.load(open('../../data/processedData/pid_s2idT1.pkl','rb'))
pid_s2id2 = pk.load(open('../../data/processedData/pid_s2idT2.pkl','rb'))

pid_s2id =  {**pid_s2id0, **pid_s2id1, **pid_s2id2}

matched_s2id_set = set()
for k,v in pid_s2id.items():
    matched_s2id_set.add(str(v))

print('len(matched_s2id_set)',len(matched_s2id_set))
pk.dump(matched_s2id_set, open('../../data/processedData/matched_s2id_set.pkl', 'wb'))
pk.dump(pid_s2id, open('../../data/processedData/pid_s2id.pkl', 'wb'))

s2id_citing_cited = {}
for i in tqdm(range(10)):
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
pk.dump(s2id_citing_cited,open('../../data/processedData/s2id_citing_citedT0.pkl','wb'))