import matplotlib.pyplot as plt
import os.path
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

businessI = len([name for name in os.listdir("./BBC/business") if os.path.isfile(os.path.join("./BBC/business", name))])
entertainmentI = len(
    [name for name in os.listdir("./BBC/entertainment") if os.path.isfile(os.path.join("./BBC/entertainment", name))])
politicsI = len([name for name in os.listdir("./BBC/politics") if os.path.isfile(os.path.join("./BBC/politics", name))])
sportI = len([name for name in os.listdir("./BBC/sport") if os.path.isfile(os.path.join("./BBC/sport", name))])
techI = len([name for name in os.listdir("./BBC/tech") if os.path.isfile(os.path.join("./BBC/tech", name))])

names = ['business', 'entertainment', 'politics', 'sport', 'tech']
values = [businessI, entertainmentI, politicsI, sportI, techI]
print(values)

plt.figure()
plt.plot()
plt.title('BBC distribution')
plt.bar(names, values)

plt.savefig('BBC-distribution.pdf')

corpus = datasets.load_files('./BBC', encoding='latin1')

x = corpus.data
y = corpus.target
vec = CountVectorizer()
matrix = vec.fit_transform(x)

trainX, testX, trainY, testY = train_test_split(matrix, y, train_size=0.8, test_size=0.2, random_state=None)
clf = MultinomialNB().fit(trainX, trainY)

predict = clf.predict(testX)
cm = confusion_matrix(testY, predict)
cr = classification_report(testY, predict, target_names=names)
acc = accuracy_score(testY, predict)
f1 = f1_score(testY, predict, average=None)

##do prior probability on test set (?)

vocabulary = len(vec.vocabulary_)




