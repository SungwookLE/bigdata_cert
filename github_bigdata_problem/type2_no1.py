################################################################################################################################################
'''
[데이콘] 와인품질분류
https://dacon.io/competitions/open/235610/overview/description
'''
################################################################################################################################################

from matplotlib.pyplot import sca
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/wine/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/wine/test.csv')
sub = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/wine/sample_submission.csv')

print(train.shape)
print(test.shape)

train.drop('index', axis=1, inplace=True)
test.drop('index', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
labeler = LabelEncoder()
labeler.fit(train["type"])
train["type"] = labeler.transform(train["type"])
test["type"] = labeler.transform(test["type"])
print(labeler.classes_)

y_train = train[["quality"]]
x_train = train.drop("quality", axis=1)
x_test = test

print(x_train.head())
print(x_test.head())

numerical_feature_columns = x_train.select_dtypes(exclude='object').columns
print(numerical_feature_columns)

#categorical_feature_columns = x_train.select_dtypes(include='object').columns
#print(categorical_feature_columns)

from scipy.stats import skew
import numpy as np

x_all = pd.concat([x_train, x_test], axis=0)
print(x_all.apply(skew).apply(abs).sort_values(ascending=False))
x_all = x_all.apply(lambda x: np.log1p(x) if abs(skew(x))>50 else x)
print(x_all.apply(skew).apply(abs).sort_values(ascending=False))

x_train = x_all[:len(x_train)]
x_test = x_all[len(x_train):]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import cross_val_score, train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train)
y_train= y_train["quality"].ravel()
y_val = y_val["quality"].ravel()


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
score =cross_val_score(clf, x_train, y_train, cv=3, scoring='accuracy')
print(np.mean(score))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
score =cross_val_score(clf, x_train, y_train, cv=3, scoring='accuracy')
print(np.mean(score))




from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(clf, param_grid={"max_depth": [5,10,15,20,30], 'min_samples_split': [10, 30,35]}, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)

clf = RandomForestClassifier(max_depth=grid.best_params_["max_depth"], min_samples_split= grid.best_params_["min_samples_split"])
score =cross_val_score(clf, x_train, y_train, cv=3, scoring='accuracy')
print(np.mean(score))

clf.fit(x_train, y_train)
from sklearn.metrics import classification_report
score = classification_report(y_val, clf.predict(x_val))
print(score)

pred = clf.predict(x_test)
ans = pd.DataFrame({"index": test.index, "pred": pred})
ans.to_csv("./github_bigdata_problem/type2_no1.csv", index=False)