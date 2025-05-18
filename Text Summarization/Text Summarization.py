# We need summarization

import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer



def summarizeText(text,featurizer):
    sentences = nltk.sent_tokenize(text)
    X = featurizer.fit_transform(sentences)
    scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        score = getSentenceScore(X[i,:])
        scores[i] = score

    print("\nSummarized Text:")
    sortIdx = np.argsort(-scores)
    for i in sortIdx[:5]:
        print(wrap("%.2f : %s" % (scores[i],sentences[i])))


def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)


def getSentenceScore(tfidf_row):
    x = tfidf_row[tfidf_row != 0]
    return x.mean()

def main():
    print("Text Summarization")
    print("Extractive Summarization")
    nltk.download('punkt')
    nltk.download('stopwords')

    df = pd.read_csv("../Datasets/bbc_text_cls.csv")
    print(df.head())

    doc = df[df.labels == 'business']['text'].sample(random_state = 42)
    print(wrap(doc.iloc[0]))


    sentences = nltk.sent_tokenize(doc.iloc[0].split("\n",1)[1])

    featurizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        norm='l1',
    )

    X = featurizer.fit_transform(sentences)

    scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        score = getSentenceScore(X[i,:])
        scores[i] = score

    sortIdx = np.argsort(-scores)

    print("\nGenerated Summary:")
    for i in sortIdx[:5]:
        print(wrap("%.2f: %s" % (scores[i],sentences[i])))


    print(doc.iloc[0].split("\n",1)[0])

    doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
    summarizeText(doc.iloc[0].split("\n",1)[1],featurizer)

    print(doc.iloc[0].split("\n,1")[0])






if __name__ == "__main__":
    main()




