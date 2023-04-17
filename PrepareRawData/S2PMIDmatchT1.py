# 1 号进程
# nohup python -u S2PMIDmatchT1.py > S2PMIDmatchT1.log 2>&1 &
# 把PMID和S2ID联系起来，只要breast cancer里的,约32万篇文章，321216
import jsonlines
from tqdm import tqdm
import pickle as pk

# read all papers
PID_set = set(pk.load(open('../../data/processedData/allPaperset.pkl','rb')))
pid_s2id1 = {}
for i in tqdm(range(10,20)):
    with jsonlines.open('/home/dell/kd_paper_data/data/SemanticScholar-20220913/full(v20220913)/papers/paperP%d.jsonl'%i, mode='r') as reader:
        for item in reader:
            if item['externalids']['PubMed'] != None and item['externalids']['PubMed'] in PID_set:
                pid_s2id1[item['externalids']['PubMed']] = item['corpusid']
print('# matched-t1',len(pid_s2id1))
pk.dump(pid_s2id1,open('../../data/processedData/pid_s2idT1.pkl','wb'))