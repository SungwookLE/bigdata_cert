################################################################################################################################################
# [데이콘] 축구선수 이적료
################################################################################################################################################
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/fifa/FIFA_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/fifa/FIFA_test.csv')
sub = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/fifa/submission.csv')

# label 설정 및 불필요 컬럼 삭제
label = train['value']
train.drop(['id', 'name', 'value'], axis=1, inplace=True)
test.drop(['id', 'name'], axis=1, inplace=True)

x_all = pd.concat([train, test], axis=0)
print(x_all)

print(x_all.isnull().sum())


categorical = x_all.select_dtypes(include='object').columns
numerical = x_all.select_dtypes(exclude='object').columns

x_categrical = pd.get_dummies(x_all[categorical])
x_all.drop(categorical, axis=1, inplace=True)
x_all = pd.concat([x_all, x_categrical], axis=1)

from scipy.stats import skew
print(x_all[numerical].apply(skew).apply(abs).sort_values(ascending=False))

import numpy as np
x_all[numerical] = x_all[numerical].apply(lambda x: np.log1p(x) if abs(skew(x))> 0.5 else x)
print(x_all[numerical].apply(skew).apply(abs).sort_values(ascending=False))

x_train = x_all[:len(train)]
x_test = x_all[len(train):]
y_train = label

from sklearn.preprocessing import StandardScaler, scale
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import PredefinedSplit, train_test_split, cross_val_score
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor()
score = cross_val_score(reg, x_train, y_train, cv=3, scoring="r2")
#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print(np.mean(score))

reg.fit(x_train, y_train)
print(reg.score(x_val, y_val))

pred = reg.predict(x_test)
sub['value']= pred
sub.to_csv("github_bigdata_problem/type2_no2.csv", index=False)

pred = reg.predict(x_train)
print(pred/y_train*100)