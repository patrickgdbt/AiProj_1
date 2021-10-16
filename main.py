import matplotlib.pyplot as plt
import os.path
import math
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

##Create distribution graph of classes

businessI = len([name for name in os.listdir("./BBC/business") if os.path.isfile(os.path.join("./BBC/business", name))])
entertainmentI = len(
    [name for name in os.listdir("./BBC/entertainment") if os.path.isfile(os.path.join("./BBC/entertainment", name))])
politicsI = len([name for name in os.listdir("./BBC/politics") if os.path.isfile(os.path.join("./BBC/politics", name))])
sportI = len([name for name in os.listdir("./BBC/sport") if os.path.isfile(os.path.join("./BBC/sport", name))])
techI = len([name for name in os.listdir("./BBC/tech") if os.path.isfile(os.path.join("./BBC/tech", name))])

names = ['business', 'entertainment', 'politics', 'sport', 'tech']
values = [businessI, entertainmentI, politicsI, sportI, techI]

plt.figure()
plt.plot()
plt.title('BBC distribution')
plt.bar(names, values)
plt.savefig('BBC-distribution.pdf')

##Load corpus

corpus = datasets.load_files('./BBC', encoding='latin1')

##Pre process

x = corpus.data
y = corpus.target
vec = CountVectorizer()
matrix = vec.fit_transform(x)

##Train test

trainX, testX, trainY, testY = train_test_split(matrix, y, train_size=0.8, test_size=0.2, random_state=None)

f = open("bbc-performance.txt", "w")
f.write("###################################################################\n")
f.write("               Multinomial default values, try 1\n")
f.write("###################################################################\n")
f.write("\n")

clf = MultinomialNB().fit(trainX, trainY)

predict = clf.predict(testX)

cm = confusion_matrix(testY, predict)

f.write("(b): confusion matrix\n")
f.write(str(cm))
f.write("\n")
f.write("\n")

cr = classification_report(testY, predict, target_names=names)

f.write("(c): classification report\n")
f.write(str(cr))
f.write("\n")
f.write("\n")

acc = accuracy_score(testY, predict)
macrof1 = f1_score(testY, predict, average='macro')
weightedf1 = f1_score(testY, predict, average='weighted')

f.write("(d): accuracy, macro f1, weighted f1\n")
f.write("accuracy: " + str(acc) + "\n")
f.write("macro f1: " + str(macrof1) + "\n")
f.write("weighted f1: " + str(weightedf1) + "\n")
f.write("\n")

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

f.write("(e): prior probabilities\n")
f.write("Business: " + businessPercentage + " \n")
f.write("Entertainment: " + entertainmentPercentage + " \n")
f.write("Politics: " + politicsPercentage + " \n")
f.write("Sport: " + sportPercentage + " \n")
f.write("Tech: " + techPercentage + " \n")
f.write("\n")

f.write("(f): size of vocabulary \n")

vocabularySize = len(vec.vocabulary_)

f.write("Vocabulary size: " + str(vocabularySize) + "\n")
f.write("\n")

businessWordTokens = matrix[y == 0].sum()
entertainmentWordTokens = matrix[y == 1].sum()
politicsWordTokens = matrix[y == 2].sum()
sportWordTokens = matrix[y == 3].sum()
techWordTokens = matrix[y == 4].sum()

f.write("(g): number of word tokens in each class")
f.write("Business: " + str(businessWordTokens) + " \n")
f.write("Entertainment: " + str(entertainmentWordTokens) + " \n")
f.write("Politics: " + str(politicsWordTokens) + " \n")
f.write("Sport: " + str(sportWordTokens) + " \n")
f.write("Tech: " + str(techWordTokens) + " \n")
f.write("\n")

entireCorpusWordTokens = businessWordTokens + entertainmentWordTokens + politicsWordTokens + sportWordTokens + techWordTokens


f.write("(h): number of word tokens in entire corpus\n")
f.write("Number of word tokens in corpus: " + str(entireCorpusWordTokens) + "\n")
f.write("\n")

indexedZeroCounts = []
for i in range(5):
    wordTokensCountAsArray = matrix[y == i].toarray()
    zeroCount = 0
    for x in wordTokensCountAsArray.T:
        if x.sum() == 0:
            zeroCount += 1
    indexedZeroCounts.append(zeroCount)

f.write("(i): number and percentage of words with frequency 0 per class\n")
f.write("Business: " + str(indexedZeroCounts[0]) + " " + str("{:.2%}".format(indexedZeroCounts[0]/len(vec.vocabulary_))) + "\n")
f.write("Entertainment: " + str(indexedZeroCounts[1]) + " " + str("{:.2%}".format(indexedZeroCounts[1]/len(vec.vocabulary_))) + "\n")
f.write("Politics: " + str(indexedZeroCounts[2]) + " " + str("{:.2%}".format(indexedZeroCounts[2]/len(vec.vocabulary_))) + "\n")
f.write("Sport: " + str(indexedZeroCounts[3]) + " " + str("{:.2%}".format(indexedZeroCounts[3]/len(vec.vocabulary_))) + "\n")
f.write("Tech: " + str(indexedZeroCounts[4]) + " " + str("{:.2%}".format(indexedZeroCounts[4]/len(vec.vocabulary_))) + "\n")
f.write("\n")

wordTokensCountAsArray = matrix.toarray()
oneCount = 0
for x in wordTokensCountAsArray.T:
    if x.sum() == 1:
        oneCount += 1

f.write("(j): number and percentage of words with frequency of 1 in the entire corpus\n")
f.write("Number: " + str(oneCount) + ", Percentage: " + str("{:.2%}".format(oneCount/len(vec.vocabulary_))) + "\n")
f.write("\n")
##people = id 19740
##american = id 2334

peopleCount = wordTokensCountAsArray.T[19740].sum()
americanCount = wordTokensCountAsArray.T[2334].sum()

f.write("(k): log probability of 2 favourite words (people, american)\n")
f.write("People: " + str(-math.log(peopleCount/vocabularySize)) + "\n")
f.write("American: " + str(-math.log(americanCount/vocabularySize)) + "\n")
f.write("\n")


f.write("\n\n\n###################################################################\n")
f.write("               Multinomial default values, try 2\n")
f.write("###################################################################\n")
f.write("\n")

clf = MultinomialNB().fit(trainX, trainY)

predict = clf.predict(testX)

cm = confusion_matrix(testY, predict)

f.write("(b): confusion matrix\n")
f.write(str(cm))
f.write("\n")
f.write("\n")

cr = classification_report(testY, predict, target_names=names)

f.write("(c): classification report\n")
f.write(str(cr))
f.write("\n")
f.write("\n")

acc = accuracy_score(testY, predict)
macrof1 = f1_score(testY, predict, average='macro')
weightedf1 = f1_score(testY, predict, average='weighted')

f.write("(d): accuracy, macro f1, weighted f1\n")
f.write("accuracy: " + str(acc) + "\n")
f.write("macro f1: " + str(macrof1) + "\n")
f.write("weighted f1: " + str(weightedf1) + "\n")
f.write("\n")

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

f.write("(e): prior probabilities\n")
f.write("Business: " + businessPercentage + " \n")
f.write("Entertainment: " + entertainmentPercentage + " \n")
f.write("Politics: " + politicsPercentage + " \n")
f.write("Sport: " + sportPercentage + " \n")
f.write("Tech: " + techPercentage + " \n")
f.write("\n")

f.write("(f): size of vocabulary \n")

vocabularySize = len(vec.vocabulary_)

f.write("Vocabulary size: " + str(vocabularySize) + "\n")
f.write("\n")

businessWordTokens = matrix[y == 0].sum()
entertainmentWordTokens = matrix[y == 1].sum()
politicsWordTokens = matrix[y == 2].sum()
sportWordTokens = matrix[y == 3].sum()
techWordTokens = matrix[y == 4].sum()

f.write("(g): number of word tokens in each class")
f.write("Business: " + str(businessWordTokens) + " \n")
f.write("Entertainment: " + str(entertainmentWordTokens) + " \n")
f.write("Politics: " + str(politicsWordTokens) + " \n")
f.write("Sport: " + str(sportWordTokens) + " \n")
f.write("Tech: " + str(techWordTokens) + " \n")
f.write("\n")

entireCorpusWordTokens = businessWordTokens + entertainmentWordTokens + politicsWordTokens + sportWordTokens + techWordTokens


f.write("(h): number of word tokens in entire corpus\n")
f.write("Number of word tokens in corpus: " + str(entireCorpusWordTokens) + "\n")
f.write("\n")

indexedZeroCounts = []
for i in range(5):
    wordTokensCountAsArray = matrix[y == i].toarray()
    zeroCount = 0
    for x in wordTokensCountAsArray.T:
        if x.sum() == 0:
            zeroCount += 1
    indexedZeroCounts.append(zeroCount)

f.write("(i): number and percentage of words with frequency 0 per class\n")
f.write("Business: " + str(indexedZeroCounts[0]) + " " + str("{:.2%}".format(indexedZeroCounts[0]/len(vec.vocabulary_))) + "\n")
f.write("Entertainment: " + str(indexedZeroCounts[1]) + " " + str("{:.2%}".format(indexedZeroCounts[1]/len(vec.vocabulary_))) + "\n")
f.write("Politics: " + str(indexedZeroCounts[2]) + " " + str("{:.2%}".format(indexedZeroCounts[2]/len(vec.vocabulary_))) + "\n")
f.write("Sport: " + str(indexedZeroCounts[3]) + " " + str("{:.2%}".format(indexedZeroCounts[3]/len(vec.vocabulary_))) + "\n")
f.write("Tech: " + str(indexedZeroCounts[4]) + " " + str("{:.2%}".format(indexedZeroCounts[4]/len(vec.vocabulary_))) + "\n")
f.write("\n")

wordTokensCountAsArray = matrix.toarray()
oneCount = 0
for x in wordTokensCountAsArray.T:
    if x.sum() == 1:
        oneCount += 1

f.write("(j): number and percentage of words with frequency of 1 in the entire corpus\n")
f.write("Number: " + str(oneCount) + ", Percentage: " + str("{:.2%}".format(oneCount/len(vec.vocabulary_))) + "\n")
f.write("\n")

##people = id 19740
##american = id 2334

peopleCount = wordTokensCountAsArray.T[19740].sum()
americanCount = wordTokensCountAsArray.T[2334].sum()

f.write("(k): log probability of 2 favourite words (people, american)\n")
f.write("People: " + str(-math.log(peopleCount/vocabularySize)) + "\n")
f.write("American: " + str(-math.log(americanCount/vocabularySize)) + "\n")
f.write("\n")


f.write("\n\n\n###################################################################\n")
f.write("     Multinomial default values, try 3 with smoothing = 0.001\n")
f.write("###################################################################\n")
f.write("\n")

clf = MultinomialNB(alpha=0.0001).fit(trainX, trainY)

predict = clf.predict(testX)

cm = confusion_matrix(testY, predict)

f.write("(b): confusion matrix\n")
f.write(str(cm))
f.write("\n")
f.write("\n")

cr = classification_report(testY, predict, target_names=names)

f.write("(c): classification report\n")
f.write(str(cr))
f.write("\n")
f.write("\n")

acc = accuracy_score(testY, predict)
macrof1 = f1_score(testY, predict, average='macro')
weightedf1 = f1_score(testY, predict, average='weighted')

f.write("(d): accuracy, macro f1, weighted f1\n")
f.write("accuracy: " + str(acc) + "\n")
f.write("macro f1: " + str(macrof1) + "\n")
f.write("weighted f1: " + str(weightedf1) + "\n")
f.write("\n")

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

f.write("(e): prior probabilities\n")
f.write("Business: " + businessPercentage + " \n")
f.write("Entertainment: " + entertainmentPercentage + " \n")
f.write("Politics: " + politicsPercentage + " \n")
f.write("Sport: " + sportPercentage + " \n")
f.write("Tech: " + techPercentage + " \n")
f.write("\n")

f.write("(f): size of vocabulary \n")

vocabularySize = len(vec.vocabulary_)

f.write("Vocabulary size: " + str(vocabularySize) + "\n")
f.write("\n")

businessWordTokens = matrix[y == 0].sum()
entertainmentWordTokens = matrix[y == 1].sum()
politicsWordTokens = matrix[y == 2].sum()
sportWordTokens = matrix[y == 3].sum()
techWordTokens = matrix[y == 4].sum()

f.write("(g): number of word tokens in each class")
f.write("Business: " + str(businessWordTokens) + " \n")
f.write("Entertainment: " + str(entertainmentWordTokens) + " \n")
f.write("Politics: " + str(politicsWordTokens) + " \n")
f.write("Sport: " + str(sportWordTokens) + " \n")
f.write("Tech: " + str(techWordTokens) + " \n")
f.write("\n")

entireCorpusWordTokens = businessWordTokens + entertainmentWordTokens + politicsWordTokens + sportWordTokens + techWordTokens

f.write("(h): number of word tokens in entire corpus\n")
f.write("Number of word tokens in corpus: " + str(entireCorpusWordTokens) + "\n")
f.write("\n")

indexedZeroCounts = []
for i in range(5):
    wordTokensCountAsArray = matrix[y == i].toarray()
    zeroCount = 0
    for x in wordTokensCountAsArray.T:
        if x.sum() == 0:
            zeroCount += 1
    indexedZeroCounts.append(zeroCount)

f.write("(i): number and percentage of words with frequency 0 per class\n")
f.write("Business: " + str(indexedZeroCounts[0]) + " " + str("{:.2%}".format(indexedZeroCounts[0]/len(vec.vocabulary_))) + "\n")
f.write("Entertainment: " + str(indexedZeroCounts[1]) + " " + str("{:.2%}".format(indexedZeroCounts[1]/len(vec.vocabulary_))) + "\n")
f.write("Politics: " + str(indexedZeroCounts[2]) + " " + str("{:.2%}".format(indexedZeroCounts[2]/len(vec.vocabulary_))) + "\n")
f.write("Sport: " + str(indexedZeroCounts[3]) + " " + str("{:.2%}".format(indexedZeroCounts[3]/len(vec.vocabulary_))) + "\n")
f.write("Tech: " + str(indexedZeroCounts[4]) + " " + str("{:.2%}".format(indexedZeroCounts[4]/len(vec.vocabulary_))) + "\n")
f.write("\n")

wordTokensCountAsArray = matrix.toarray()
oneCount = 0
for x in wordTokensCountAsArray.T:
    if x.sum() == 1:
        oneCount += 1

f.write("(j): number and percentage of words with frequency of 1 in the entire corpus\n")
f.write("Number: " + str(oneCount) + ", Percentage: " + str("{:.2%}".format(oneCount/len(vec.vocabulary_))) + "\n")
f.write("\n")

##people = id 19740
##american = id 2334

peopleCount = wordTokensCountAsArray.T[19740].sum()
americanCount = wordTokensCountAsArray.T[2334].sum()

f.write("(k): log probability of 2 favourite words (people, american)\n")
f.write("People: " + str(-math.log(peopleCount/vocabularySize)) + "\n")
f.write("American: " + str(-math.log(americanCount/vocabularySize)) + "\n")
f.write("\n")


f.write("\n\n\n###################################################################\n")
f.write("      Multinomial default values, try 4 with smoothing = 0.9\n")
f.write("###################################################################\n")
f.write("\n")

clf = MultinomialNB(alpha=0.9).fit(trainX, trainY)

predict = clf.predict(testX)

cm = confusion_matrix(testY, predict)

f.write("(b): confusion matrix\n")
f.write(str(cm))
f.write("\n")
f.write("\n")

cr = classification_report(testY, predict, target_names=names)

f.write("(c): classification report\n")
f.write(str(cr))
f.write("\n")
f.write("\n")

acc = accuracy_score(testY, predict)
macrof1 = f1_score(testY, predict, average='macro')
weightedf1 = f1_score(testY, predict, average='weighted')

f.write("(d): accuracy, macro f1, weighted f1\n")
f.write("accuracy: " + str(acc) + "\n")
f.write("macro f1: " + str(macrof1) + "\n")
f.write("weighted f1: " + str(weightedf1) + "\n")
f.write("\n")

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

f.write("(e): prior probabilities\n")
f.write("Business: " + businessPercentage + " \n")
f.write("Entertainment: " + entertainmentPercentage + " \n")
f.write("Politics: " + politicsPercentage + " \n")
f.write("Sport: " + sportPercentage + " \n")
f.write("Tech: " + techPercentage + " \n")
f.write("\n")

f.write("(f): size of vocabulary \n")

vocabularySize = len(vec.vocabulary_)

f.write("Vocabulary size: " + str(vocabularySize) + "\n")
f.write("\n")

businessWordTokens = matrix[y == 0].sum()
entertainmentWordTokens = matrix[y == 1].sum()
politicsWordTokens = matrix[y == 2].sum()
sportWordTokens = matrix[y == 3].sum()
techWordTokens = matrix[y == 4].sum()

f.write("(g): number of word tokens in each class")
f.write("Business: " + str(businessWordTokens) + " \n")
f.write("Entertainment: " + str(entertainmentWordTokens) + " \n")
f.write("Politics: " + str(politicsWordTokens) + " \n")
f.write("Sport: " + str(sportWordTokens) + " \n")
f.write("Tech: " + str(techWordTokens) + " \n")
f.write("\n")

entireCorpusWordTokens = businessWordTokens + entertainmentWordTokens + politicsWordTokens + sportWordTokens + techWordTokens

f.write("(h): number of word tokens in entire corpus\n")
f.write("Number of word tokens in corpus: " + str(entireCorpusWordTokens) + "\n")
f.write("\n")

indexedZeroCounts = []
for i in range(5):
    wordTokensCountAsArray = matrix[y == i].toarray()
    zeroCount = 0
    for x in wordTokensCountAsArray.T:
        if x.sum() == 0:
            zeroCount += 1
    indexedZeroCounts.append(zeroCount)

f.write("(i): number and percentage of words with frequency 0 per class\n")
f.write("Business: " + str(indexedZeroCounts[0]) + " " + str("{:.2%}".format(indexedZeroCounts[0]/len(vec.vocabulary_))) + "\n")
f.write("Entertainment: " + str(indexedZeroCounts[1]) + " " + str("{:.2%}".format(indexedZeroCounts[1]/len(vec.vocabulary_))) + "\n")
f.write("Politics: " + str(indexedZeroCounts[2]) + " " + str("{:.2%}".format(indexedZeroCounts[2]/len(vec.vocabulary_))) + "\n")
f.write("Sport: " + str(indexedZeroCounts[3]) + " " + str("{:.2%}".format(indexedZeroCounts[3]/len(vec.vocabulary_))) + "\n")
f.write("Tech: " + str(indexedZeroCounts[4]) + " " + str("{:.2%}".format(indexedZeroCounts[4]/len(vec.vocabulary_))) + "\n")
f.write("\n")

wordTokensCountAsArray = matrix.toarray()
oneCount = 0
for x in wordTokensCountAsArray.T:
    if x.sum() == 1:
        oneCount += 1

f.write("(j): number and percentage of words with frequency of 1 in the entire corpus\n")
f.write("Number: " + str(oneCount) + ", Percentage: " + str("{:.2%}".format(oneCount/len(vec.vocabulary_))) + "\n")
f.write("\n")

##people = id 19740
##american = id 2334

peopleCount = wordTokensCountAsArray.T[19740].sum()
americanCount = wordTokensCountAsArray.T[2334].sum()

f.write("(k): log probability of 2 favourite words (people, american)\n")
f.write("People: " + str(-math.log(peopleCount/vocabularySize)) + "\n")
f.write("American: " + str(-math.log(americanCount/vocabularySize)) + "\n")
f.write("\n")