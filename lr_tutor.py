# this problem is from a Kaggle competition, link: https://www.kaggle.com/competitions/tutors-lessons-prices-prediction/
# you can download training data via opendatasets library

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

train = pd.read_excel('tutors-lessons-prices-prediction/train.xlsx')
test = pd.read_excel('tutors-lessons-prices-prediction/test.xlsx')
submit = pd.read_csv('tutors-lessons-prices-prediction/sample_submit.csv')

# sns.boxplot(train['mean_price'], width=0.4)
# plt.show()

pd.set_option('display.max_columns', None)

# предмет - либо математика, либо информатика
train = pd.get_dummies(train, columns=['предмет'])

# уровень тьютора - аспирант, студент, частный препод и т д
tmp = train['status'].str.get_dummies(sep=',')
train = train.drop('status', axis=1)
train = pd.concat([train, tmp], axis=1)

# парсим опыт
train['experience'] = train['experience'].str.replace(r"[^\d]", "", regex=True)
train['experience'] = train['experience'].astype('float64')
train['experience'] = train['experience'].fillna(0)


def strtolist(q):
    q = q[1:-1]
    q = q.split("', '")
    q[0] = q[0][1:]
    q[-1] = q[-1][:-1]
    return q

def parse_lists(data, column):
    data[column] = data[column].apply(strtolist)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data[column])
    tmp = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    data = pd.concat([data, tmp], axis=1)
    data = data.drop(column, axis=1)
    return data

# TODO: тут миллиард колонок, можно разбить, например, на две - индивидуально/пара/группа, дошкольники/школьники/студенты
train = parse_lists(train, 'categories') # кем является
# TODO: поудаляй ненужные, ну или выдели в special
train = parse_lists(train, 'tutor_head_tags') # к чему готовит
print(train[train[''] != 0])
train = train.drop('', axis=1) # ничего не указал

# удаляем ненужные столбцы; TODO: с description можно просто есть/нет
train = train.drop('Unnamed: 0', axis=1)
train = train.drop('ФИО', axis=1)

print(train.info())
print(train['GMAT (математическая часть)'].value_counts())