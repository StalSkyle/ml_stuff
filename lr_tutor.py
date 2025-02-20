# this problem is from a Kaggle competition, link: https://www.kaggle.com/competitions/tutors-lessons-prices-prediction/
# you can download training data via opendatasets library

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_excel('tutors-lessons-prices-prediction/train.xlsx')
test = pd.read_excel('tutors-lessons-prices-prediction/test.xlsx')
submit = pd.read_csv('tutors-lessons-prices-prediction/sample_submit.csv')

pd.set_option('display.max_columns', None)

# FEATURE ENGINEERING

'''
кратко что сделано: кодирование нечисловых признаков - предмет, status, experience, categories, tutor_head_tags
description и experience_des - либо есть либо нет
дропнуты ненужные признаки
предмет - либо математика, либо информатика
образование, ученая степень и ученое звание - объеденено в educations, +1 за образование, +5 за степень, +3 за звание, -0.5, если нет описания образования
ВОЗМОЖНЫЕ УЛУЧШЕНИЯ: 
-миллиард колонок categories, можно разбить, например, на две - индивидуально/пара/группа, дошкольники/школьники/студенты
-tutor_head_tags: поудаляй ненужные, ну или выдели в отдельную колонку special
мб сделать образование, ученые степень и звание разными колонками
можно распарсить год окончания, университет, пед/не пед, специальность и квалификацию
сделать, чтобы прибавлялось в зависимости от звания
УБРАТЬ ВЫБРОСЫ!11!!111!!111
нормализовать/стандартизировать
'''


train = pd.get_dummies(train, columns=['предмет'])

tmp = train['status'].str.get_dummies(sep=',')
train = train.drop('status', axis=1)
train = pd.concat([train, tmp], axis=1)

train['experience'] = train['experience'].str.replace(r"[^\d]", "", regex=True)
train['experience'] = train['experience'].astype('float64')
train['experience'] = train['experience'].fillna(0)


def strtolist(q):
    q = q[1:-1]
    q = q.split("', '")
    q[0] = q[0][1:]
    q[-1] = q[-1][:-1]
    return q


# зачем fittransform, если можно getdummies - мб перепиши
def parse_lists(data, column):
    data[column] = data[column].apply(strtolist)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data[column])
    q = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    data = pd.concat([data, q], axis=1)
    data = data.drop(column, axis=1)
    return data


train = parse_lists(train, 'categories')  # кем является

train = parse_lists(train, 'tutor_head_tags')  # к чему готовит
train = train.drop('', axis=1)  # ничего не указал, таких всего 5

train['description'] = train['description'].apply(
    lambda x: 0 if pd.isna(x) or isinstance(x,
                                            str) and 'Репетитор не предоставил о себе дополнительных сведений' in x else 1)
train['experience_desc'] = train['experience_desc'].apply(
    lambda x: 0 if pd.isna(x) or isinstance(x,
                                            str) and 'Репетитор не предоставил о себе дополнительных сведений' in x else 1)

train = train.drop('Unnamed: 0', axis=1)
train = train.drop('ФИО', axis=1)

train['tutor_rating'] = train['tutor_rating'].fillna(0)

train['educations'] = 0
for i in range(1, 7):
    train[f'Education_{i}'] = train[f'Education_{i}'].apply(
        lambda x: 0 if pd.isna(x) else 1)
    train[f'Desc_Education_{i}'] = train[f'Desc_Education_{i}'].apply(
        lambda x: 0 if pd.isna(x) else 1)
    train['educations'] += train[f'Education_{i}'] - 0.5 * (
            (train[f'Desc_Education_{i}'] == 0) * (
        train[f'Education_{i}']))
    train = train.drop([f'Education_{i}'], axis=1)
    train = train.drop([f'Desc_Education_{i}'], axis=1)

for i in range(1, 3):  # something something dont repeat yourself
    train[f'Ученое звание {i}'] = train[f'Ученое звание {i}'].apply(
        lambda x: 0 if pd.isna(x) else 1)
    train['educations'] += 5 * train[f'Ученое звание {i}']
    train[f'Ученая степень {i}'] = train[f'Ученая степень {i}'].apply(
        lambda x: 0 if pd.isna(x) else 1)
    train['educations'] += 3 * train[f'Ученая степень {i}']
    train = train.drop(f'Ученая степень {i}', axis=1)
    train = train.drop(f'Ученое звание {i}', axis=1)

# train = train[train['mean_price'] < 40]

# ОБУЧЕНИЕ

# sns.set_theme(rc = {'figure.figsize':(30, 30)})
#
# sns.heatmap(train.corr(), annot = False, cmap="YlGnBu", linecolor='white', linewidths=1)
# plt.show()

X = train.drop(columns=['mean_price'])
Y = train['mean_price']

# смотрим на оценки
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size = 0.2)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

Y_pred = lin_reg.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', mean_squared_error(Y_test, Y_pred))
print('R2 score:', r2_score(Y_test, Y_pred))


# lin_reg = LinearRegression()
# lin_reg.fit(X, Y)

