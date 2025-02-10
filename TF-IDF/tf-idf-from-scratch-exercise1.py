#Exercise: use CountVectorizer to form the counts instead.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

df = pd.read_csv('../bbc_text_cls.csv')
print(df.head())

vectorizer = CountVectorizer(lowercase=True,token_pattern=r'\b\w+\b')
X_counts = vectorizer.fit_transform(df['text'])

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

word2idx = vectorizer.vocabulary_
idx2word = {v:k for k,v in word2idx.items()}

N= X_tfidf.shape[0]
V = len(vectorizer.get_feature_names_out())

np.random.seed(223)
i = np.random.choice(N)
row = df.iloc[i]

print("\nLabel:",row['labels'])
print("Text:",row['text'].split("\n",1)[0])
print("\nTop 5 terms:")

scores = X_tfidf[i].toarray()[0]
indices = np.argsort(-scores)

for j in indices[:5]:
    print(idx2word[j])

