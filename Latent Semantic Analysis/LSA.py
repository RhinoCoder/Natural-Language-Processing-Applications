import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD




def Tokenizer(s):

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    wordnetLemmatizer = WordNetLemmatizer()


    stops = set(stopwords.words('english'))
    stops = stops.union({
        'introduction', 'edition', 'series', 'application',
        'approach', 'card', 'access', 'package', 'plus', 'etext',
        'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
        'third', 'second', 'fourth', 'volume'})

    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)

    #Removing short words as they are probably not useful/important
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnetLemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stops]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

    return tokens

def main():
    dataset = "../Datasets/all_book_titles.txt"
    titles = [line.rstrip() for line in open(dataset)]
    print("Okunan başlık sayısı:", len(titles))

    vectorizer = CountVectorizer(binary=True,tokenizer=Tokenizer)
    X = vectorizer.fit_transform(titles)
    print("X.shape:", X.shape)
    indexWordMap = vectorizer.get_feature_names_out()
    print("Feature sayısı:", len(indexWordMap))
    print("Örnek 10 feature:", indexWordMap[:10])
    X = X.T
    print("X.T shape:", (X.T).shape)

    svd = TruncatedSVD()
    Z = svd.fit_transform(X)

    x_coords = Z[:, 0]
    y_coords = Z[:, 1]
    plt.figure(figsize=(12, 8))
    plt.scatter(x_coords, y_coords, s=5, color="blue")
    for i, term in enumerate(indexWordMap):
        plt.text(x_coords[i], y_coords[i], term, fontsize=4)
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D SVD of Book Title Features")
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    main()