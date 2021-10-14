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

# do prior probability on test set (?)
businessN = 0
entertainmentN = 0
politicsN = 0
sportN = 0
techN = 0

for category in testY:
    if category == 0:
        businessN += 1
    elif category == 1:
        entertainmentN += 1
    elif category == 2:
        politicsN += 1
    elif category == 3:
        sportN += 1
    elif category == 4:
        techN += 1

total = len(testY)
businessPercentage = "{:.2%}".format(businessN / total)
entertainmentPercentage = "{:.2%}".format(entertainmentN / total)
politicsPercentage = "{:.2%}".format(politicsN / total)
sportPercentage = "{:.2%}".format(sportN / total)
techPercentage = "{:.2%}".format(techN / total)

print("Prior Probability:")
print("Business: " + businessPercentage)
print("Entertainment: " + entertainmentPercentage)
print("Politics: " + politicsPercentage)
print("Sport: " + sportPercentage)
print("Tech: " + techPercentage)

vocabulary = len(vec.vocabulary_)
print(vocabulary)

businessWordTokens = matrix[y == 0].sum()
entertainmentWordTokens = matrix[y == 1].sum()
politicsWordTokens = matrix[y == 2].sum()
sportWordTokens = matrix[y == 3].sum()
techWordTokens = matrix[y == 4].sum()

print(businessWordTokens)
print(entertainmentWordTokens)
print(politicsWordTokens)
print(sportWordTokens)
print(techWordTokens)

entireCorpusWordTokens = businessWordTokens + entertainmentWordTokens + politicsWordTokens + sportWordTokens + techWordTokens

print(entireCorpusWordTokens)

count = 0