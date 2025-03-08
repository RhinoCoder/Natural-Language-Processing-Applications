import numpy as np
import pandas as pd
import textwrap
import nltk

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def main():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('all')
    nltk.download('averaged_perceptron_tagger')

    df = pd.read_csv("../Datasets/bbc_text_cls.csv")
    labels = set(df['labels'])
    print("Labels are :", labels)
    label = 'business'
    texts = df[df['labels'] == label]['text']
    print(texts.head())


    probs = {}

    for doc in texts:
        lines = doc.split("\n")
        for line in lines:
            tokens = word_tokenize(line)
            for i in range(len(tokens) - 2):
                t0 = tokens[i]
                t1 = tokens[i + 1]
                t2 = tokens[i + 2]
                key = (t0, t2)
                if key not in probs:
                    probs[key] = {}
                if t1 not in probs[key]:
                    probs[key][t1] = 1
                else:
                    probs[key][t1] += 1


    for key, d in probs.items():
        total = sum(d.values())
        for k, v in d.items():
            d[k] = v / total


    detokenizer = TreebankWordDetokenizer()

    np.random.seed(1234)
    i = np.random.choice(texts.shape[0])
    doc = texts.iloc[i]
    newDoc = spinDocument(doc, probs, detokenizer)
    print(textwrap.fill(newDoc, replace_whitespace=False, fix_sentence_endings=True))


def spinDocument(doc, probs, detokenizer):
    lines = doc.split("\n")
    output = []
    for line in lines:
        if line:
            newLine = spinLine(line, probs, detokenizer)
        else:
            newLine = line
        output.append(newLine)

    return "\n".join(output)


def sampleWord(d):
    p0 = np.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t

    assert (False)


def spinLine(line, probs, detokenizer):
    tokens = word_tokenize(line)
    i = 0
    output = [tokens[0]]
    while i < (len(tokens) - 2):
        t0 = tokens[i]
        t1 = tokens[i + 1]
        t2 = tokens[i + 2]
        key = (t0, t2)
        if key in probs:
            pDist = probs[key]

            if len(pDist) > 1 and np.random.random() < 0.3:
                middle = sampleWord(pDist)
                if middle != t1:
                    output.append("<" + middle + ">")
                else:
                    output.append(t1)
                output.append(t2)
                i += 2
            else:
                output.append(t1)
                i += 1
        else:
            output.append(t1)
            i += 1

    if i < len(tokens):
        for j in range(i, len(tokens)):
            output.append(tokens[j])

    return detokenizer.detokenize(output)


if __name__ == "__main__":
    main()