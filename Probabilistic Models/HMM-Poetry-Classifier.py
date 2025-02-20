import numpy as np
import pandas as pd
import nltk
import os
import time

from sklearn.model_selection import train_test_split


robertPath = "../Datasets/robert_frost.txt"
allanPath = "../Datasets/edgar_allan_poe.txt"

allanLines = []
robertLines = []


#Let's first split lines of these text files.
def SplitIntoLines(path,usedList):
    count = 0
    with open(path, "r") as poetry:
        for line in poetry:
            usedList.append(line.strip())
            count +=1


def TokenizeText(poetry):
    allTokens = []
    for line in poetry:
        tokens = line.lower().split()
        allTokens.extend(tokens)

    return allTokens


def SplitTrainTest(indexedLines,labels,test_size = 0.2):
    XTrain,XTest,YTrain,YTest = train_test_split(
        indexedLines,labels,test_size= test_size,random_state=42
    )

    return XTrain,XTest,YTrain,YTest

def ConvertLinesToIndices(lines,word2index):
    indexedLines = []
    for line in lines:
        tokens = line.lower().split()
        indices = [word2index.get(word,word2index["<UNK>"]) for word in tokens]
        indexedLines.append(indices)

    return indexedLines



def AnimatedDots(message,delay = 0.15):
    print(message,end="",flush=True)
    for i in range(15):
        time.sleep(delay)
        print(".",end="",flush=True)
    print("\n")


def main():

    AnimatedDots("Splitting into Lines")
    SplitIntoLines(allanPath,allanLines)
    SplitIntoLines(robertPath,robertLines)

    AnimatedDots("Tokenizing Lines")
    combinedVocabulary = allanLines + robertLines
    labels = [0] * len(allanLines) + [1] * len(robertLines)
    allTokens = TokenizeText(combinedVocabulary)

    #Map each unique word to a unique integer value.
    uniqueWords = set(allTokens)
    word2index = {word: idx for idx,word in enumerate(uniqueWords,start=0)}
    word2index["<UNK>"] = len(word2index)

    AnimatedDots("Converting lines into indices")
    indexedLines = ConvertLinesToIndices(combinedVocabulary,word2index)

    #Split data into test & train set.
    XTrain, XTest, YTrain, YTest = SplitTrainTest(indexedLines,labels)

if __name__ == "__main__":
    main()
