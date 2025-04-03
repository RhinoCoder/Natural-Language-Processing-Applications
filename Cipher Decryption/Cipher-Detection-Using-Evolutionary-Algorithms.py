import string
import random
import re
import requests
import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt




def updateTransition(ch1,ch2,MM):
    # ASCII representation transformation
    i = ord(ch1) - 97
    j = ord(ch2) - 97
    MM[i,j] +=1


def updatePi(ch,pi):
    i = ord(ch) - 97
    pi[i] += 1


def getWordProbability(word,pi,MM):
    i = ord(word[0]) - 97
    logP = np.log(pi[i])

    for ch in word[1:]:
        j = ord(ch) - 97
        logP += np.log(MM[i,j])
        i = j


    return logP

def getSequenceProbs(words,pi,MM):
    #Kinda tokenization for processing words'chars.
    if type(words) == str:
        words = words.split()

    logP = 0
    for word in words:
        logP += getWordProbability(word,pi,MM)


    return logP


def encodeMessage(msg,regex,trueMapping):
    msg = msg.lower()
    msg = regex.sub(' ',msg)
    codedMsg = []
    for ch in msg:
        codedCh = ch
        if ch in trueMapping:
            codedCh = trueMapping[ch]
        codedMsg.append(codedCh)

    return ''.join(codedMsg)

def decodeMessage(msg,wordMap):

    decodedMessage = []
    for ch in msg:
        decodedCh = ch
        if ch in wordMap:
            decodedCh = wordMap[ch]
        decodedMessage.append(decodedCh)

    return ''.join(decodedMessage)


def evolveOffspring(dnaPool,nChildren):
    offspring = []
    for dna in dnaPool:
        for _ in range(nChildren):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))

            temp = copy[j]
            copy[j] = copy[k]
            copy[k] = temp
            offspring.append(copy)


    return offspring + dnaPool

def main():

    mobyDick ="../Datasets/moby_dick.txt"
    if not os.path.exists(mobyDick):
        print("Moby dick.txt dataset is not found!,"
              "You can search for its name on google & gutenberg project to get the data in plain text format.")
    else:
        print("Dataset Exists, yo ho ho")

    np.random.seed(42)
    letters1 = list(string.ascii_lowercase)
    letters2 = list(string.ascii_lowercase)
    trueMapping = {}
    random.shuffle(letters2)

    for key, val in zip(letters1, letters2):
        trueMapping[key] = val

    MarkovMatrix = np.ones((26, 26))
    pInitial = np.zeros(26)

    regex = re.compile('[^a-zA-Z]')
    for line in open(mobyDick):
        line = line.rstrip()

        if line:
            line = regex.sub(' ',line)
            tokens = line.lower().split()

            for token in tokens:
                ch0 = token[0]
                updatePi(ch0,pInitial)

                for ch1 in token[1:]:
                    updateTransition(ch0,ch1,MarkovMatrix)
                    ch0 = ch1

    pInitial /= pInitial.sum()
    MarkovMatrix /= MarkovMatrix.sum(axis=1,keepdims=True)

    originalMessage = '''I then lounged down the street and found, as I expected, that there
    was a mews in a lane which runs down by one wall of the garden. I lent
    the ostlers a hand in rubbing down their horses, and received in
    exchange twopence, a glass of half-and-half, two fills of shag tobacco,
    and as much information as I could desire about Miss Adler, to say
    nothing of half a dozen other people in the neighbourhood in whom I was
    not in the least interested, but whose biographies I was compelled to
    listen to.'''

    encodedMessage = encodeMessage(originalMessage,regex,trueMapping)

    dnaPool = []
    for _ in range(20):
        dna = list(string.ascii_lowercase)
        random.shuffle(dna)
        dnaPool.append(dna)


    numIters = 1000
    scores = np.zeros(numIters)
    bestDna = None
    bestMap = None
    bestScore = float("-inf")

    for i in range(numIters):
        if i > 0:
            dnaPool = evolveOffspring(dnaPool,3)

        dna2score = {}
        for dna in dnaPool:
            currentMap = {}
            for key,val in zip(letters1,dna):
                currentMap[key] = val

            decodedMessage = decodeMessage(encodedMessage,currentMap)
            score = getSequenceProbs(decodedMessage,pInitial,MarkovMatrix)
            dna2score[''.join(dna)] = score

            if score > bestScore:
                bestDna = dna
                bestMap = currentMap
                bestScore = score

        scores[i] = np.mean(list(dna2score.values()))
        sortedDna = sorted(dna2score.items(),key=lambda x:x[1],reverse=True)
        dnaPool = [list(k) for k,v in sortedDna[:5]]

        if i % 100 == 0:
            print(f"Iteration:{i}, Score:{scores[i]}, Best so far: {bestScore}")




    decodedMessage = decodeMessage(encodedMessage,bestMap)
    print(f"LL of decoded message:{getSequenceProbs(decodedMessage,pInitial,MarkovMatrix)}")
    print(f"LL of true message:{getSequenceProbs(regex.sub(' ',originalMessage.lower()),pInitial,MarkovMatrix)}")

    for true,v in trueMapping.items():
        pred = bestMap[v]
        if true != pred:
            print(f"Actual: {true}, Predicted:{pred}")


    print("Decoded message:\n",textwrap.fill(decodedMessage))
    print("True message:\n",originalMessage)

    plt.plot(scores)
    plt.show()

if __name__ == "__main__":
    main()