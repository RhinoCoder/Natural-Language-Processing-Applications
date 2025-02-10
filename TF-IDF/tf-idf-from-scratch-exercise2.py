# Exercise(hard): use Scipy's csr_matrix instead.
# You can not use X[i,j] += 1 here.

import pandas as pd
import numpy as np
import nltk

from nltk import word_tokenize
from scipy.sparse import coo_matrix,csr_matrix

nltk.download('all')

df = pd.read_csv('../bbc_text_cls.csv')

#Populate word to index mapping matrix.
idx = 0
word2idx = {}
tokenized_docs = []

for doc in df['text']:
    words = word_tokenize(doc.lower())
    doc_as_int = []

    for word in words:
        if word not in word2idx:
            word2idx[word] = idx
            idx +=1

        doc_as_int.append(word2idx[word])

    tokenized_docs.append(doc_as_int)


#Making reverse mapping -> index to word
idx2word = {v: k for k,v in word2idx.items()}

#Number of documents and vocabulary size.
N = len(df['text'])
V = len(word2idx)

#Build a sparse term-frequency matrix using COO format
rows = []
cols = []
values = []

for i,doc_as_int in enumerate(tokenized_docs):
    for j in doc_as_int:
        rows.append(i) # Document index
        cols.append(j) # Word index
        values.append(1) # Term frequency

#Create COO sparse matrix and convert to CSR for efficient computation
tf = coo_matrix((values,(rows,cols)),shape = (N,V)).tocsr()

document_freq = np.array(tf.astype(bool).sum(axis=0)).flatten()
idf = np.log(N /(document_freq + 1))
tf_idf = tf.multiply(idf).tocsr()

np.random.seed(183)
i = np.random.choice(N)
row = df.iloc[i]

print("\nLabel:",row['labels'])
print("Text:",row['text'].split("\n",1)[0]+"\n")
print("Top 5 terms: ")

scores = tf_idf[i].toarray()[0]
indices = np.argsort(-scores)

for j in indices[:5]:
    print(idx2word[j])
