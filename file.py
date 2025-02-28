import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

train = pd.read_csv('./star-type-classification/train_star.csv')
test = pd.read_csv('./star-type-classification/test_star.csv')

le = LabelEncoder()
train['TargetClass'] = le.fit_transform(train['TargetClass'])


# парсим sptype
train['SpType'] = train['SpType'].str.lower()

sptypes = ['o', 'b', 'a', 'f', 'g', 'k', 'm']
for i in range(len(sptypes)):
    train[sptypes[i]] = 0

for i in range(len(sptypes)):
    train[sptypes[i]] = train['SpType'].str.contains(sptypes[i]).astype(int)

lumin = ['ia', 'iab', 'ib', 'ii', 'iii', 'iv', 'v']
for i in range(len(lumin)):
    train[lumin[i]] = 0

for i in range(1, len(sptypes)):
    if lumin[i] != 'ii':
        train[lumin[i]] = train['SpType'].str.contains(lumin[i]).astype(int)

train['ia'] = (train['SpType'].str.contains('ia') & (~(train['SpType'].str.contains('iab')))).astype(int)
train['ii'] = (train['SpType'].str.contains('ii') & (~(train['SpType'].str.contains('iii')))).astype(int)

train = train.drop('SpType', axis=1)
print(train.head())

plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='YlGnBu')
plt.show()