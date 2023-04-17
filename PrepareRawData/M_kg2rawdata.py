# nohup python -u M_kg2rawdata.py > kg2rawdata.log 2>&1 &
import operator
from tqdm import tqdm



# generate paper-author mapping
f = open("../../data/paper_author.csv", "r")
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
print(len(author_paper))
print(len(paper_author))