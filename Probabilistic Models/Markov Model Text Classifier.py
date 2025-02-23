import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score


def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                pi[idx] += 1
            else:
                A[last_idx, idx] += 1

            last_idx = idx


class Classifier:

    def __init__(self,logAs,logpis,logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors)

    def _compute_log_likelihood(self,input_,class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        lastidx = None
        logprob = 0

        for idx in input_:
            if lastidx is None:
                logprob += logpi[idx]
            else:
                logprob += logA[lastidx,idx]

            lastidx = idx

        return logprob

    def predict(self,inputs):
        predictions = np.zeros(len(inputs))

        for i,input_ in enumerate(inputs):
            posteriors = [self._compute_log_likelihood(input_,c) + self.logpriors[c] \
                          for c in range(self.K)]
            pred = np.argmax(posteriors)
            predictions[i] = pred

        return predictions



def main():


    input_files = [
        '../Datasets/edgar_allan_poe.txt',
        '../Datasets/robert_frost.txt'
    ]

    input_texts = []
    labels = []

    for label, f in enumerate(input_files):
        print(f"{f} corresponds to labels {label}")

        for line in open(f):
            # rstrip() removes new line.
            line = line.rstrip().lower()
            if line:
                line = line.translate(str.maketrans('', '', string.punctuation))
                input_texts.append(line)
                labels.append(label)

    train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels)


    print("\nDebugging purposes")
    print(".......................")
    print("Len of Ytrain:",len(Ytrain))
    print("Len of Ytest:",len(Ytest))
    print("train text first 3 entry:",train_text[:3])
    print("Ytrain first 3 entry:",Ytrain[:3])

    idx = 1
    word2idx = {'<unk>': 0}

    # Populate word 2 idx
    for text in train_text:
        tokens = text.split()
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

    train_text_int = []
    test_text_int = []

    for text in train_text:
        tokens = text.split()
        line_as_int = [word2idx[token] for token in tokens]
        train_text_int.append(line_as_int)

    for text in test_text:
        tokens = text.split()
        line_as_int = [word2idx.get(token, 0) for token in tokens]
        test_text_int.append(line_as_int)



    # Initialize A and pi matrices - for both classes

    V = len(word2idx)
    A0 = np.ones((V, V))
    pi0 = np.ones(V)

    A1 = np.ones((V, V))
    pi1 = np.ones(V)

    compute_counts([t for t,y in zip(train_text_int,Ytrain) if y == 0],A0,pi0)
    compute_counts([t for t,y in zip(train_text_int,Ytrain) if y == 1],A1,pi1)

    A0 /= A0.sum(axis=1,keepdims=True)
    pi0 /= pi0.sum()

    A1 /= A1.sum(axis=1, keepdims=True)
    pi1 /= pi1.sum()

    logA0 = np.log(A0)
    logpi0 = np.log(pi0)

    logA1 = np.log(A1)
    logpi1 = np.log(pi1)

    #Compute priors
    count0 = sum( y == 0 for y in Ytrain)
    count1 = sum(y == 1 for y in Ytrain)
    total = len(Ytrain)
    p0 = count0 / total
    p1 = count1 / total
    logp0 = np.log(p0)
    logp1 = np.log(p1)

    print("Priors",p0,p1,logp0,logp1)
    print(".......................")

    clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

    Ptrain = clf.predict(train_text_int)
    print(f"\nTrain accuracy: {np.mean(Ptrain == Ytrain)}")

    Ptest = clf.predict(test_text_int)
    print(f"Test accuracy: {np.mean(Ptest == Ytest)}")

    conf_matrx = confusion_matrix(Ytrain, Ptrain)
    print("\nConfusion Matrix:\n ", conf_matrx)

    f1scoreTrain = f1_score(Ytrain,Ptrain)
    print("\nF1 score train: ",f1scoreTrain)

    f1scoreTest = f1_score(Ytest,Ptest)
    print("F1 score test: ", f1scoreTest)


if __name__ == "__main__":
    main()
