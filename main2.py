import statistics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Load the CSV into a Pandas Data Frame
df = pd.read_csv('drug200.csv')

# Group data
grouped_by_drug = df.groupby(["Drug"])
grouped_by_drug.size().plot.bar()

# Save distribution to PDF
plt.savefig('drug_dist.pdf')

"""
Gender is nominal.
BP and Cholesterol are ordinal.

*Make sure that your converted format respects the ordering of ordinal features, 
and does not introduce any ordering for nominal features.*

Hence, gender will use get_dummies as this does not introduce ordering.
BP and Cholesterol need to have order preserved (LOW = 0 etc)
I'm not sure if we need to assign a strict int, 
but the column is "Category" and not just a string (for BP and Chol) and we can change to int easily.

If we want to go from category to int theres also better ways to do it that the assignment doesn't reference.
"""
categories = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
# df['Drug'] = pd.Categorical(df['Drug'], categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
df2 = pd.get_dummies(df, columns=['Sex'])

df2['BP'] = pd.Categorical(df2['BP'], categories=['LOW', 'NORMAL', 'HIGH'])
df2['BP'] = df2['BP'].cat.codes

df2['Cholesterol'] = pd.Categorical(df2['Cholesterol'], categories=['LOW', 'NORMAL', 'HIGH'])
df2['Cholesterol'] = df2['Cholesterol'].cat.codes

# Uncomment if we want to assign class to category (unsure).
# df['Drug'] = pd.Categorical(df['Drug'], categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])

# x, y = df2.copy().drop(['Drug']), df2['Drug']
y = df2['Drug']
x = df2.drop(['Drug'], axis=1)
train_x, test_x, train_y, test_y = train_test_split(x, y)

models = [ MultinomialNB(), DecisionTreeClassifier(), Perceptron(), MLPClassifier() ]

f = open("drugs-performance.txt", "w")
f.write("Step 7:\n")

for model in models:
    f.write("--------------------------------------------\n")
    f.write("a) "+str(type(model))+" with default parameters\n")
    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    cm = confusion_matrix(test_y, predict, labels=categories)
    f.write("b) Confusion Matrix:\n ")
    f.write(str(cm)+"\n")
    cr = classification_report(test_y, predict)
    f.write("c)\n")
    f.write(str(cr))
    f.write("d)\n")
    acc = accuracy_score(test_y, predict)
    f.write("Accuracy: " + str(acc)+"\n")
    f1Macro = f1_score(test_y, predict, average='macro')
    f.write("F1 Macro Average: " + str(f1Macro)+"\n")
    f1Weighted = f1_score(test_y, predict, average='weighted')
    f.write("F1 Weighted Average: " + str(f1Weighted)+"\n")

f.write("Step 8:\n")
for model in models:
    accuracy = []
    f1Macro = []
    f1Weighted = []
    for i in range(10):
        model.fit(train_x, train_y)
        predict = model.predict(test_x)
        accuracy.append(accuracy_score(test_y, predict))
        f1Macro.append(f1_score(test_y, predict, average='macro'))
        f1Weighted.append(f1_score(test_y, predict, average='weighted'))
    f.write("-------------------------------------------\n")
    f.write(str(type(model))+" with default parameters\n")
    f.write("Average Accuracy: " + str(statistics.mean(accuracy))+"\n")
    f.write("Average F1 Macro Average: " + str(statistics.mean(f1Macro))+"\n")
    f.write("Average F1 Weighted Average: " + str(statistics.mean(f1Weighted))+"\n")
    f.write("Standard Deviation Accuracy: " + str(statistics.stdev(accuracy))+"\n")
    f.write("Standard Deviation F1 Macro Average: " + str(statistics.stdev(accuracy))+"\n")
    f.write("Standard Deviation F1 Weighted Average: " + str(statistics.stdev(accuracy))+"\n")

f.close()
