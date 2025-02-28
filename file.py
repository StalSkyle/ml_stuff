import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

train = pd.read_csv('./star-type-classification/train_star.csv')
test = pd.read_csv('./star-type-classification/test_star.csv')

le = LabelEncoder()
train['TargetClass'] = le.fit_transform(train['TargetClass'])


# распарсить sptype
train['SpType'] = train['SpType'].str.lower()

sptypes = ['o', 'b', 'a', 'f', 'g', 'k', 'm']
for i in range(len(sptypes)):
    train[sptypes[i]] = 0

lumin = ['ia', 'iab', 'ib', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
for i in range(len(lumin)):
    train[lumin[i]] = 0

print(train.head())