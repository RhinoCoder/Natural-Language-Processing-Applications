import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize

nltk.download('all')

df = pd.read_csv('../Datasets/bbc_text_cls.csv')
df.head()

#Populate word2idx
#Convert documents into sequences of ints / ids / indices

idx = 0
word2idx ={}
tokenized_docs = []

for doc in df['text']:
  words = word_tokenize(doc.lower())
  doc_as_int = []

  for word in words:
    if word not in word2idx:
      word2idx[word] = idx
      idx += 1

    #Save for later use.
    doc_as_int.append(word2idx[word])
  tokenized_docs.append(doc_as_int)


#Reverse mapping, if you do it smarter you can store it as a list
idx2word = {v:k for k,v in word2idx.items()}

#Number of documets
N = len(df['text'])

#Number of Words
V = len(word2idx)

#Instantiate term-frequency matrix
#Note: Could have also used count vectorizer
tf = np.zeros((N,V))

#Populate term frequency counts

for i,doc_as_int in enumerate(tokenized_docs):
  for j in doc_as_int:
    tf[i,j] +=1


#Compute IDF
document_freq = np.sum(tf > 0,axis=0) #Document frequency (shape = (V,))
idf = np.log(N / document_freq)

#Computer TF-IDF
tf_idf = tf * idf
np.random.seed(123)

#Pick a random document, show the top 5 terms ( in terms of tf_idf score)
i = np.random.choice(N)
row = df.iloc[i]
print("Label:",row['labels'])
print("Text:",row['text'].split("\n",1)[0])
print("Top 5 terms:")

scores = tf_idf[i]
indices = (-scores).argsort()

for j in indices[:5]:
  print(idx2word[j])




