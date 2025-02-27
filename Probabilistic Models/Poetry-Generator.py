import string
import re
import time

import numpy as np



def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))


def addToDict(d,k,v):
    if k not in d:
        d[k] = []
    d[k].append(v)


def list2pdict(ts):
    #Turn each possibilities into a dictionary of probabilities
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t,0.) + 1
    for t,c in d.items():
        d[t] = c / n
    return d


def sample_word(d):
    p0 = np.random.random()
    cumulative = 0
    for t,p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t
    assert(False) # Should never get there.


def generate(initial,first_order,second_order):
    print("Generating The Poem......\n")
    for i in range(4):
        sentence = []
        #Initial word
        w0 = sample_word(initial)
        sentence.append(w0)

        #Second word
        w1 = sample_word(first_order[w0])
        sentence.append(w1)

        #Second order transitions until END
        while True:
            w2 = sample_word(second_order[(w0,w1)])
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2

        print(' '.join(sentence))

def main():

    randomSeedForNp = int(time.time())
    inputTextFile = "../Datasets/robert_frost.txt"
    np.random.seed(randomSeedForNp)
    initial = {}
    first_order = {}
    second_order = {}

    for line in open(inputTextFile):
        tokens = remove_punctuation(line.rstrip().lower()).split()
        T = len(tokens)
        for i in range(T):
            t = tokens[i]
            if i == 0:
                # Measure the distribution of the first word
                initial[t] = initial.get(t,0.) + 1
            else:
                t_1 = tokens[i-1]
                if i == T - 1:
                    addToDict(second_order,(t_1,t),'END')
                if i == 1:
                    # Measure distribution of second word
                    # Given only first word
                    addToDict(first_order,t_1,t)
                else:
                    t_2 = tokens[i-2]
                    addToDict(second_order,(t_2,t_1),t)


    initial_total = sum(initial.values())
    for t,c in initial.items():
        initial[t] = c / initial_total


    for t_1,ts in first_order.items():
        first_order[t_1] = list2pdict(ts)

    for k,ts in second_order.items():
        second_order[k] = list2pdict(ts)


    generate(initial,first_order,second_order)


if __name__ == "__main__":
    main()