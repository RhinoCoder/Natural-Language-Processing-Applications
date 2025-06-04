import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import textwrap

from nltk.corpus import  stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import  NMF



def plotTopWords(model,featureNames,nTopWords = 10):
    fig,axes = plt.subplots(2,5,figsize =(25,15),sharex=True)
    axes = axes.flatten()

    for topicIdx,topic in enumerate(model.components_):
        topFeaturesInd = topic.argsort()[: -nTopWords -1 : -1]
        topFeatures = [featureNames[i] for i in topFeaturesInd]
        weights = topic[topFeaturesInd]

        ax = axes[topicIdx]
        ax.barh(topFeatures,weights,height=0.7)
        ax.set_title(f"Topic {topicIdx + 1}", fontdict = {"fontsize" : 25})
        ax.invert_yaxis()
        ax.tick_params(axis = "both",which = "major", labelsize = 12)

        for i in "top right left".split():
            ax.spines[i].set_visible(False)

        fig.suptitle('LDA',fontsize=25)

    plt.subplots_adjust(top = 0.80,bottom = 0.05 , wspace = 0.95, hspace = 0.35)
    plt.show()




def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)



def main():
    print("NMF started")
    nltk.download('stopwords')
    stops = set(stopwords.words('english'))

    stops = stops.union({
        'said', 'would', 'could', 'told', 'also', 'one', 'two', 'mr', 'new', 'year',
    })
    stops = list(stops)
    df = pd.read_csv("../Datasets/bbc_text_cls.csv")
    print(df.head())

    vectorizer = TfidfVectorizer(stop_words=stops)

    X = vectorizer.fit_transform(df['text'])

    nmf = NMF(n_components=10,
              beta_loss="kullback-leibler",
              solver='mu',
              random_state=0,
    )

    nmf.fit(X)

    featureNames = vectorizer.get_feature_names_out()
    plotTopWords(nmf,featureNames)
    Z= nmf.transform(X)

    np.random.seed(0)
    i = np.random.choice(len(df))
    z = Z[i]
    topics = np.arange(10) + 1

    fig,ax = plt.subplots()
    ax.barh(topics,z)
    ax.set_yticks(topics)
    ax.set_title('True label: %s' % df.iloc[i]['labels'])
    plt.show()
    print(wrap(df.iloc[i]['text']))


if __name__ == "__main__":
    main()