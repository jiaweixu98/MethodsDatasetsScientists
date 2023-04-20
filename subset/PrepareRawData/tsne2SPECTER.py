# nohup python -u tsne2SPECTER.py > tsne2SPECTER.log 2>&1 &
# 这tsne的复杂度太高了，所以我们用PCA，希望可以奏效。
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk


s2id_embeddings0 = pk.load(open('../../data/processedData/s2id_embeddings0.pkl', 'rb'))
s2id_embeddings1 = pk.load(open('../../data/processedData/s2id_embeddings1.pkl', 'rb'))

s2id_embeddings = {**s2id_embeddings0, **s2id_embeddings1}


X_train = []
X_train_names = []
for x in s2id_embeddings:
        X_train.append(s2id_embeddings[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
pca_embeddings = {}
# embedding_file = open('pca_embedding_SPECTER.txt', 'w')

# PCA with 128 dimensions.
pca =  PCA(n_components = 128)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
# U1 = pca.components_
# print(X_fit)
for i, x in enumerate(X_train_names):
        pca_embeddings[x] = list(X_fit[i])
        # embedding_file.write("%s\t" % x)
        # for t in pca_embeddings[x]:
        #         embedding_file.write("%f\t" % t)        
        # embedding_file.write("\n")
print('len(pca_embeddings)',len(pca_embeddings))
pk.dump(pca_embeddings,open('../../data/processedData/pca_embeddings.pkl','wb'))