from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(
    '../Datasets/GoogleNews-vectors-negative300.bin',
    binary=True
)

#w1 - w2 = ? - w3
def find_analogies(w1,w2,w3):
    r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
    print(" %s - %s = %s - %s" % (w1,w2,r[0][0],w3))



#Examples can be incremented and added/provided more.
find_analogies('king','man','woman')
find_analogies('france','paris','london')
find_analogies('france','paris','rome')


def nearest_neighbors(w):
    r = word_vectors.most_similar(positive=[w])
    print("Neighbors of: %s" % w)

    for word,score in r:
        print("\t%s" % word)





nearest_neighbors('king')
nearest_neighbors('france')
