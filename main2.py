import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
# df['Drug'] = pd.Categorical(df['Drug'], categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
df2 = pd.get_dummies(df, columns=['Sex'])

df2['BP'] = pd.Categorical(df2['BP'], categories=['LOW', 'NORMAL', 'HIGH'])
df2['Cholesterol'] = pd.Categorical(df2['Cholesterol'], categories=['LOW', 'NORMAL', 'HIGH'])
# Uncomment if we want to assign class to category (unsure).
# df['Drug'] = pd.Categorical(df['Drug'], categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])

train, test = train_test_split(df2)