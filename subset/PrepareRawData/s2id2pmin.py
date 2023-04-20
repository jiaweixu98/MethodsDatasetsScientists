import pickle as pk
# load s2id keys
# pca_embeddings = pk.load(open('../../data/processedData/pca_embeddings.pkl','rb'))
# pmid_pca_embeddings = {}
# print('len(pca_embeddings)', len(pca_embeddings))
s2id_citing_cited = pk.load(open('../../data/processedData/s2id_citing_cited.pkl','rb'))
pmid_citing_cited = {}
print('len(s2id_citing_cited)', len(s2id_citing_cited))


pid_s2id = pk.load(open('../../data/processedData/pid_s2id.pkl', 'rb'))

s2id_pid = {}

for k, v in pid_s2id.items():
    s2id_pid[str(v)] = k

# pk.dump(s2id_pid,open('../../data/processedData/s2id_pid.pkl', 'wb'))

# for k,v in pca_embeddings.items():
#     pmid_pca_embeddings[s2id_pid[str(k)]] = v
# print('len(pmid_pca_embeddings)', len(pmid_pca_embeddings))

# pk.dump(pmid_pca_embeddings,open('../../data/processedData/pmid_pca_embeddings.pkl', 'wb'))

for k, v in s2id_citing_cited.items():
    pmid_citing_cited[s2id_pid[k]] = []
    for ref in v:
        pmid_citing_cited[s2id_pid[k]].append(s2id_pid[ref])

print('len(pmid_citing_cited)', len(pmid_citing_cited))

pk.dump(pmid_citing_cited, open('../../data/processedData/pmid_citing_cited.pkl', 'wb'))
