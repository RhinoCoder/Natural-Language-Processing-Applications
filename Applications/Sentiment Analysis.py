import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_array_ops import const


def plotConfusionMatrix(cm):
    classes = ['negative','positive','neutral']
    dfCm = pd.DataFrame(cm,index=classes,columns=classes)
    ax = sn.heatmap(dfCm,annot=True,fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    plt.show()





def main():
    np.random.seed(1)
    dataPath = "../Datasets/AirlineTweets.csv"
    dfI = pd.read_csv(dataPath)
    print(dfI.head())


    df = dfI[['airline_sentiment','text']].copy()
    print(df.head())

    targetMap = {'positive' : 1,'negative' : 0,'neutral' : 2}
    df['target'] = df['airline_sentiment'].map(targetMap)
    print(df.head())

    dfTrain,dfTest = train_test_split(df)

    print(f"Train shape: {dfTrain.shape}")
    print(f"Test shape: {dfTest.shape}")

    df['airline_sentiment'].hist()
    plt.show()

    vectorizer = TfidfVectorizer(max_features=2000)
    model = LogisticRegression(max_iter=500)


    xTrain = vectorizer.fit_transform(dfTrain['text'])
    xTest = vectorizer.transform(dfTest['text'])

    yTrain = dfTrain['target']
    yTest = dfTest['target']

    model.fit(xTrain,yTrain)
    print(f"Train accuracy: {model.score(xTrain, yTrain)}")
    print(f"Test accuracy: {model.score(xTest, yTest)}")

    prTrain = model.predict_proba(xTrain)
    prTest = model.predict_proba(xTest)
    print(f"Training AUC: {roc_auc_score(yTrain, prTrain,multi_class='ovo')}")
    print(f"Test AUC: {roc_auc_score(yTest, prTest,multi_class='ovo')}")

    pTrain = model.predict(xTrain)
    pTest = model.predict(xTest)


    confMat = confusion_matrix(yTrain,pTrain,normalize='true')
    print(confMat)
    plotConfusionMatrix(confMat)

    confTest = confusion_matrix(yTest,pTest,normalize='true')
    plotConfusionMatrix(confTest)

    binaryTargetList = [targetMap['positive'],targetMap['negative']]
    dfBTrain = dfTrain[dfTrain['target'].isin(binaryTargetList)]
    dfBTest = dfTest[dfTest['target'].isin(binaryTargetList)]
    print(dfBTrain.head)

    xTrain = vectorizer.fit_transform(dfBTrain['text'])
    xTest = vectorizer.transform(dfBTest['text'])
    yTrain = dfBTrain['target']
    yTest = dfBTest['target']

    model = LogisticRegression(max_iter=500)
    model.fit(xTrain,yTrain)
    print(f"Train accuracy: {model.score(xTrain, yTrain)}")
    print(f"Test accuracy: {model.score(xTest, yTest)}")

    prTrain = model.predict_proba(xTrain)[:,1]
    prTest = model.predict_proba(xTest)[:,1]
    print(f"Training AUC: {roc_auc_score(yTrain, prTrain)}")
    print(f"Test AUC: {roc_auc_score(yTest, prTest)}")

    print(model.coef_)
    plt.hist(model.coef_[0],bins=30)
    plt.show()

    wordIndexMap = vectorizer.vocabulary_
    print(wordIndexMap)

    chosenRange = 2
    print("Most positive words: ")
    for word,index in wordIndexMap.items():
        weight = model.coef_[0][index]
        if weight > chosenRange:
            print(word,weight)


    print("Most negative words: ")
    for word,index in wordIndexMap.items():
        weight = model.coef_[0][index]
        if weight < -chosenRange:
            print(word,weight)


#Exercise print the most-wrong tweets for both classes.
# Find a negative review where p( y=1 | X) is closest to 1
# Find a positive review where p( y=1 | X) is closest to 0
# Set class_weight='balanced'
if __name__ == "__main__":
    main()








