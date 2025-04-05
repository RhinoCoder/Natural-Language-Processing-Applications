#Naive bayes P(Y | X ) = P(X | Y) P(Y) / SIGMA(y -> n) (X | y) P(y)
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from tensorflow.python.eager.monitoring import Counter
from wordcloud import WordCloud


#Data set link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
def plotConfusionMatrix(cm):
    classes = ['ham','spam']
    df_cm = pd.DataFrame(cm,index=classes,columns=classes)
    ax = sn.heatmap(df_cm,annot=True,fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    plt.show()


def visualize(label,df):
    words = ''
    for msg in df[df['labels']== label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600,height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()




def main():
    dataSetPath = "../Datasets/spam.csv"
    df = pd.read_csv(dataSetPath, encoding='ISO-8859-1')
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df.columns = ['labels', 'data']

    print(df)
    print(df['labels'].hist())

    df['b_labels'] = df['labels'].map({'ham' : 0, 'spam' : 1})
    Y = df['b_labels'].to_numpy()
    dfTrain,dfTest,yTrain,yTest = train_test_split(df['data'],Y,test_size=0.3)

    #Can be used TfidfVectorizer(decode_error='ignore')
    featurizer = CountVectorizer(decode_error='ignore')
    xTrain = featurizer.fit_transform(dfTrain)
    xTest = featurizer.transform(dfTest)
    print(xTrain.shape)

    model = MultinomialNB()
    model.fit(xTrain,yTrain)
    print(f"Train accuracy: {model.score(xTrain,yTrain)}")
    print(f"Test accuracy: {model.score(xTest, yTest)}")

    pTrain = model.predict(xTrain)
    pTest = model.predict(xTest)
    print(f"Train F1 Score: {f1_score(yTrain,pTrain)}")
    print(f"Test F1 Score: {f1_score(yTest, pTest)}")

    probTrain = model.predict_proba(xTrain)[:,1]
    probTest = model.predict_proba(xTest)[:,1]
    print(f"Training AUC: {roc_auc_score(yTrain,probTrain)}")
    print(f"Test AUC: {roc_auc_score(yTest,probTest)}")

    confusionMatrix = confusion_matrix(yTrain,pTrain)
    print(confusionMatrix)
    plotConfusionMatrix(confusionMatrix)

    confusionMatrixTest = confusion_matrix(yTest,pTest)
    plotConfusionMatrix(confusionMatrixTest)

    visualize('spam',df)
    visualize('ham',df)


    X = featurizer.transform(df['data'])
    df['predictions'] = model.predict(X)

    sneakySpam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
    for msg in sneakySpam:
        print(f"Sneaky spams:{msg}")

    notSpam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
    for msg in notSpam:
        print(f"Not actually spam:{msg}")


if __name__ == "__main__":
    main()





















