# Exercise: download pretrained GloVe vectors from
# https://nlp.stanford.edu/projects/glove/
# Implement your own find_analogies() and nearest_neighbors()
# Hint: you do NOT have to go hunting around on Stackoverflow
# you do NOT have to copy and paste/
# @authorRhinoCoder
import numpy as np
from numpy.linalg import norm


def LoadGloveModel(glove_text_file):
    glove_dict = {}
    with open(glove_text_file,"r",encoding="utf-8") as inp:
        for line in inp:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:],dtype=np.float32)
            glove_dict[word] = vector
    print("Successfully loaded the model..")
    return glove_dict



def cosine_similarity(vec1,vec2):
        sim = np.dot(vec1,vec2) / (norm(vec1)*norm(vec2))
        return sim

def FindAnalogies(model,word1,word2,word3):

    if word1 not in model or word2 not in model or word3 not in model:
        return None

    similarity = -1
    vec1 = model[word1]
    vec2 = model[word2]
    vec3 = model[word3]
    resultingVector = vec2 - vec1 + vec3
    bestMatch = None
    bestScore = -float("inf")

    for word,vec in model.items():
        if word in {word1,word2,word3}:
            continue
        similarity = cosine_similarity(resultingVector,vec)
        if similarity > bestScore:
            bestScore = similarity
            bestMatch = word

    print(f"'{word1}' is to '{word2}' as '{word3}' is to '{bestMatch}'")



def NearestNeighbors(model,word,top_n = 5):
    if word not in model:
        print("Word does not exist in the model.")
        return []

    vectorWord = model[word]
    similarities = []

    for otherWord,vec in model.items():
        if otherWord == word:
            continue
        similarity = cosine_similarity(vectorWord,vec)
        similarities.append((otherWord,similarity))
    similarities.sort(key=lambda x: x[1],reverse=True)
    return similarities[:top_n]



def main():

    #Provide your model path to the program.
    glove_text_file_path = "../Datasets/glove.6B.300d.txt"

    #Load the given embeddings.
    glove_embeddings = LoadGloveModel(glove_text_file_path)

    print("Finding analogies of given words....")
    FindAnalogies(glove_embeddings,"man","king","woman")
    print("\n Nearest neighbor of 'queen",NearestNeighbors(glove_embeddings,"queen"))


if __name__ == "__main__":
    main()
