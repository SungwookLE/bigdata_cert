'''
[데이콘] 와인품질분류
https://dacon.io/competitions/open/235610/overview/description
'''

import pandas as pd
from sklearn import multiclass
from sklearn.utils import shuffle

train = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/wine/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/wine/test.csv')
sub = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/wine/sample_submission.csv')


label = train["quality"]
train.drop('quality', axis=1, inplace=True)

print(train)
print(test)

X_all = pd.concat([train, test], axis=0)

print(X_all.isnull().sum())
X_all.drop('index', axis=1, inplace=True)

type_map = {"white": 0, "red": 1}
X_all["type"]=X_all["type"].map(type_map).astype(int)
print(X_all.dtypes)

from scipy.stats import skew
import numpy as np
print(X_all.apply(skew).apply(abs).sort_values(ascending=False))
X_all= X_all.apply(lambda x: np.log1p(x) if abs(skew(x))>0.5 else x)
print(X_all.apply(skew).apply(abs).sort_values(ascending=False))

X_train = X_all[:len(train)]
X_test = X_all[len(train):]
y_train = label

print(y_train.value_counts())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train, y_train)
X_all = pd.DataFrame(data=scaler.transform(X_all), columns = X_all.columns)

X_train = X_all[:len(train)]
X_test = X_all[len(train):]

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(multi_class='ovr')
score = cross_val_score(clf, X_train, y_train, cv=3)
print(np.mean(score))

clf = GradientBoostingClassifier()
score = cross_val_score(clf, X_train, y_train, cv=3)
print(np.mean(score))

clf = RandomForestClassifier()
score = cross_val_score(clf, X_train, y_train, cv=3)
print(np.mean(score))

print('_'*30)
X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, shuffle=True)
clf.fit(X_train, y_train)
score = clf.score(x_val, y_val)
print(np.mean(score))


pred = clf.predict(x_val)

from sklearn.metrics import roc_auc_score, classification_report
score = roc_auc_score(y_val, clf.predict_proba(x_val), multi_class='ovr')
print(classification_report(y_val, clf.predict(x_val)))
print(score)

print('='*30)
ans = pd.DataFrame(data= pred, columns=['pred'])
ans.to_csv("github_bigdata_problem/type2_no4.csv", index=False)

ret = pd.read_csv("github_bigdata_problem/type2_no4.csv")
print(ret)