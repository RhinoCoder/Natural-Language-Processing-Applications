from collections import defaultdict

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
        indices = [word2index.get(word,word2index['<unk>']) for word in tokens]
        indexedLines.append(indices)

    return indexedLines


def TrainSecondOrderMarkovModel(lines):

    transitionCounts = defaultdict(lambda: defaultdict(int))
    pairCounts = defaultdict(int)

    for line in lines:

        if len(line) < 3:
            continue

        for i in range(len(line) - 2):
            wordPair = (line[i],line[i+1])
            nextWord = line[i + 2]
            transitionCounts[wordPair][nextWord] += 1
            pairCounts[wordPair] += 1


    #Add one smoothing
    vocabularySize = len(set([word for line in lines for word in line]))
    transitionProbabilities = defaultdict(lambda:defaultdict(float))

    for wordPair,nextWords in transitionCounts.items():
        totalTransitions = sum(nextWords.values()) + vocabularySize
        for nextWord,count in nextWords.items():
            transitionProbabilities[wordPair][nextWord] = (count + 1) / totalTransitions

        for line in lines:
            for word in line:
                if word not in nextWords:
                    transitionProbabilities[wordPair][word] = 1 / totalTransitions

    return transitionProbabilities

def AnimatedDots(message,delay = 0.1):
    print(message,end="",flush=True)
    for i in range(7):
        time.sleep(delay)
        print(".",end="",flush=True)
    print("\n")


def ComputePriors(labels):
    totalLines = len(labels)
    classCounts = defaultdict(int)

    for label in labels:
        classCounts[label] += 1

    priors = {k:v / totalLines for k,v in classCounts.items()}
    return priors

def ComputeLikelihood(line,transitionProbs):


    if len(line) < 3:
        return 0.0

    logLikelihood = 0.0

    for i in range(len(line) - 2):
        wordPair = (line[i],line[i+1])
        nextWord = line[i+2]

        if wordPair in transitionProbs and nextWord in transitionProbs[wordPair]:
            prob = transitionProbs[wordPair][nextWord]
        else:
            prob = 1e-10

        logLikelihood += np.log(prob)

    return logLikelihood


def PredictLine(line,transProbClass0,transProbClass1,priors):

    logLikelihood0 = ComputeLikelihood(line,transProbClass0)
    logLikelihood1 = ComputeLikelihood(line,transProbClass1)

    #Calculate posterior probabilities using Bayes' rule.
    logPosterior0 = logLikelihood0 + np.log(priors[0])
    logPosterior1 = logLikelihood1 + np.log(priors[1])

    #Make a prediction.
    return 0 if logPosterior0 > logPosterior1 else 1



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
    word2index['<unk>'] = len(word2index)

    AnimatedDots("Converting lines into indices")
    indexedLines = ConvertLinesToIndices(combinedVocabulary,word2index)

    #Split data into test & train set.
    XTrain, XTest, YTrain, YTest = SplitTrainTest(indexedLines,labels)

    # Train second-order Markov models for each class
    AnimatedDots("Training Second-Order Markov Models")
    class0Lines = [XTrain[i] for i in range(len(XTrain)) if YTrain[i] == 0]
    class1Lines = [XTrain[i] for i in range(len(XTrain)) if YTrain[i] == 1]

    transProbClass0 = TrainSecondOrderMarkovModel(class0Lines)
    transProbClass1 = TrainSecondOrderMarkovModel(class1Lines)

    priors = ComputePriors(labels)

    correctPredictions = 0
    totalPredictions = len(XTest)
    AnimatedDots("Predicting on test data with Second-Order Model")

    for i, line in enumerate(XTest):
        predictedClass = PredictLine(line, transProbClass0, transProbClass1, priors)
        if predictedClass == YTest[i]:
            correctPredictions += 1

    accuracy = correctPredictions / totalPredictions
    print(f"Second-Order Markov Model Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()