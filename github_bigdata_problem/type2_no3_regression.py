'''
[데이콘] 영화관객수 예측
'''

from nis import cat
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/movie/movies_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/movie/movies_test.csv')
sub = pd.read_csv('https://raw.githubusercontent.com/inrap8206/Bigdata_Analyst_Certificate_Korean/main/data/movie/submission.csv')

label = train["box_off_num"]
train.drop('box_off_num', axis=1, inplace=True)


X_all = pd.concat([train, test], axis=0)
X_all.drop(['title','distributor','director'], axis=1, inplace=True)
print(X_all)

print(X_all["dir_prev_bfnum"].mean())
X_all.fillna(X_all["dir_prev_bfnum"].mean(), inplace=True)

X_all["release_time"] = pd.to_datetime(X_all["release_time"]).dt.strftime("%Y").astype(object)
print(X_all["release_time"])

numerical = X_all.select_dtypes(exclude='object').columns
categorical = X_all.select_dtypes(include='object').columns

X_train = X_all[:len(train)]
X_test = X_all[len(train):]

from sklearn.preprocessing import LabelEncoder
for col in categorical:
    labeler = LabelEncoder()
    labeler.fit(X_train[col])
    X_all[col]=labeler.transform(X_all[col])
    

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[numerical])
X_all[numerical] = scaler.transform(X_all[numerical])

print(X_all.head())
X_train = X_all[:len(train)]
X_test = X_all[len(train):]
y_train = label

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import numpy as np



print('='*30)
reg = LinearRegression()
score0 = -cross_val_score(reg, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
score1 = cross_val_score(reg, X_train, y_train, cv=3, scoring='r2')

reg.fit(X_train, y_train)
score2 = reg.score(X_train, y_train)
print(np.mean(score0),np.mean(score1), score2)

reg = AdaBoostRegressor()
score0 = -cross_val_score(reg, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
score1 = cross_val_score(reg, X_train, y_train, cv=3, scoring='r2')

reg.fit(X_train, y_train)
score2 = reg.score(X_train, y_train)
print(np.mean(score0),np.mean(score1), score2)

reg = RandomForestRegressor(max_depth=7)
score0 = -cross_val_score(reg, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
score1 = cross_val_score(reg, X_train, y_train, cv=3, scoring='r2')
reg.fit(X_train, y_train)
score2 = reg.score(X_train, y_train)

print(np.mean(score0),np.mean(score1), score2)
############ 결정계수 스코어 기준 35% 인데, 너무 낮네.. 흠

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True)
reg.fit(X_train,y_train)
score2 = reg.score(X_val, y_val)
print(np.mean(score2)) ############ 결정계수 스코어 기준 44% 인데, 너무 낮네.. 흠

pred = reg.predict(X_test)
ans = pd.DataFrame(data=pred, columns=["pred"])
ans.to_csv("github_bigdata_problem/type2_no3.csv", index=False)

ret = pd.read_csv("github_bigdata_problem/type2_no3.csv")
print(ret)

