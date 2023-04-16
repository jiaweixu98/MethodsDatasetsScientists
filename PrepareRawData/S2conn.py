# nohup python -u S2conn.py > S2conn.log 2>&1 &
# 把PMID和S2ID联系起来，只要breast cancer里的
import jsonlines
from tqdm import tqdm
import pickle as pk

PID_set = set(pk.load(open('pmid_list_with_datasetBioAuthors.pkl','rb')))
pid_s2id = {}
for i in tqdm(range(30)):
    with jsonlines.open('/home/dell/kd_paper_data/data/SemanticScholar-20220913/full(v20220913)/papers/paperP%d.jsonl'%i, mode='r') as reader:
        # item即为每一行的数据
        for item in reader:
            if item['externalids']['PubMed'] != None and item['externalids']['PubMed'] in PID_set:
                pid_s2id[item['externalids']['PubMed']] = item['corpusid']
print('# matched',len(pid_s2id))
pk.dump(pid_s2id,open('pid_s2id.pkl','wb'))