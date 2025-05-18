import pandas as pd
import numpy as np
import textwrap
import nltk
from nltk.corpus import stopwords
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def summarizeText(text,factor = 0.15):
    sentences = nltk.sent_tokenize(text)
    featurizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        norm='l1'
    )
    X = featurizer.fit_transform(sentences)
    S = cosine_similarity(X)
    S /= S.sum(axis=1,keepdims=True)
    U = np.ones_like(S) / len(S)
    S = (1 - factor) * S + factor * U
    eigenVals,eigenVecs = np.linalg.eig(S.T)
    scores = eigenVecs[:,0] / eigenVecs[:,0].sum()
    sortIdx = np.argsort(-scores)

    for i in sortIdx[:5]:
        print(wrap("%.2f: %s" % (scores[i],sentences[i])))


def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)


def main():
    print("Text Rank Summarization")
    nltk.download('punkt')
    nltk.download('stopwords')

    df = pd.read_csv("../Datasets/bbc_text_cls.csv")
    print(df.head())

    doc = df[df.labels =='business']['text'].sample(random_state=42)
    print(wrap(doc.iloc[0]))
    print("\n")
    print(doc.iloc[0].split("\n",1)[1])


    sentences = nltk.sent_tokenize(doc.iloc[0].split("\n",1)[1])
    featurizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        norm='l1'
    )

    X = featurizer.fit_transform(sentences)
    Similarity = cosine_similarity(X)
    print(f"Shape of Similarity:{Similarity.shape}")
    print(f"Len of sentences: {len(sentences)}")

    Similarity /= Similarity.sum(axis=1,keepdims=True)
    print(f"\nSum is:{Similarity[0].sum()}")

    UniformTransitionMatrix = np.ones_like(Similarity) / len(Similarity)
    print(f"Sum of UTM: {UniformTransitionMatrix[0].sum()}")


    factor = 0.15
    Similarity = (1 - factor) * Similarity + factor * UniformTransitionMatrix
    print(f"\nSum is:{Similarity[0].sum()}")

    eigenVals,eigenVecs = np.linalg.eig(Similarity.T)
    print(eigenVals,"\n")
    print(eigenVecs[:,0],"\n")
    print(eigenVecs[:,0].dot(Similarity))
    print(eigenVecs[:,0] / eigenVecs[:,0].sum())

    limitingDist = np.ones(len(Similarity)) / len(Similarity)
    threshold = 1e-8
    delta = float('inf')
    iters = 0
    while delta > threshold:
        iters +=1
        p = limitingDist.dot(Similarity)
        delta = np.abs(p - limitingDist).sum()
        limitingDist = p

    print(f"Iters: {iters}")
    print(f"Limiting Dist: {limitingDist}")
    print(f"Sum of Limiting Dist: {limitingDist.sum()}")
    print(f"{ np.abs( eigenVecs[:,0] / eigenVecs[:,0].sum() - limitingDist).sum() }")


    scores = limitingDist
    sortIdx = np.argsort(-scores)
    print("\nGenerated Summary: ")
    for i in sortIdx[:5]:
        print(wrap("%.2f: %s" % (scores[i],sentences[i])))


    print(doc.iloc[0].split("\n")[0])

    doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
    summarizeText(doc.iloc[0].split("\n",1)[1])
    print(doc.iloc[0].split("\n")[0])

    summarizer = TextRankSummarizer()
    parser = PlaintextParser.from_string(
        doc.iloc[0].split("\n", 1)[1],
        Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count=5)

    print(f"Sumy summary is: \n {summary}")
    for s in summary:
        print(wrap(str(s)))



    print(f"Summarizer summary is: \n")
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=5)
    for s in summary:
        print(wrap(str(s)))



if __name__ == "__main__":
    main()



