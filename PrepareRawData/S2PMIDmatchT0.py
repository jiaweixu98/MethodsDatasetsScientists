# 0 号进程
# nohup python -u S2PMIDmatchT0.py > S2PMIDmatchT0.log 2>&1 &
# 把PMID和S2ID联系起来，只要breast cancer里的,约32万篇文章，321216, 318884
import jsonlines
from tqdm import tqdm
import pickle as pk

# read all papers
PID_set = set(pk.load(open('../../data/processedData/allPaperset.pkl','rb')))
pid_s2id0 = {}
for i in tqdm(range(10)):
    with jsonlines.open('/home/dell/kd_paper_data/data/SemanticScholar-20220913/full(v20220913)/papers/paperP%d.jsonl'%i, mode='r') as reader:
        for item in reader:
            if item['externalids']['PubMed'] != None and item['externalids']['PubMed'] in PID_set:
                pid_s2id0[item['externalids']['PubMed']] = item['corpusid']
print('# matched-t0',len(pid_s2id0))
pk.dump(pid_s2id0,open('../../data/processedData/pid_s2idT0.pkl','wb'))