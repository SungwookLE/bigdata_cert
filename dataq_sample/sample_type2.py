# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

################################################################################################
# https://dataq.goorm.io/exam/116674/%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/3
# 사용자 코딩
################################################################################################

print('='*30)
length_train = len(X_train)
length_test = len(X_test)

X_train.drop('cust_id', axis=1, inplace=True)
X_test_cust_id = X_test["cust_id"]
X_test.drop('cust_id', axis=1, inplace=True)
y_train.drop('cust_id', axis=1, inplace=True)

################################################################################################
# Feature Engineering
################################################################################################

################################################################################################
# 1-1. 결측값 확인 후 채우기 fillna
################################################################################################

X_all = pd.concat([X_train, X_test], axis=0)
print(X_all.isnull().sum())
print('-'*30)
X_all["환불금액"].fillna(X_all["환불금액"].mean(), inplace=True)
print(X_all.isnull().sum())
print('='*30)

################################################################################################
# 1-2. one-hot encoding
################################################################################################

numerical_columns = X_all.select_dtypes(exclude='object').columns
categorical_columns = X_all.select_dtypes(include='object').columns

print('='*30)
print(X_all[categorical_columns].head())

X_categorical = pd.get_dummies(X_all[categorical_columns])

####################################
# labelencoder를 사용한다면 아래와 같이
####################################
#from sklearn.preprocessing import LabelEncoder
#for col in categorical_columns: 
#	labeler = LabelEncoder()
#	labeler.fit(X_all[col])
#	X_all[col] = labeler.transform(X_all[col])
#	print('-'*30)
	
print(X_categorical.head())

X_all.drop(categorical_columns, axis=1, inplace=True)
X_all = pd.concat([X_all, X_categorical], axis=1)

print('='*30)

################################################################################################
# 1-3. min-max scaler: skew 없애기전에 해준 이유가 값에 음수가 있어 np.log1p 연산시 NaN이 되는 문제가 있어,
# min-max로 음수를 없앰
################################################################################################

print(X_all[numerical_columns].head())
X_train = X_all[:length_train]
X_test = X_all[length_train:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train[numerical_columns])
X_all[numerical_columns] = scaler.transform(X_all[numerical_columns])
print('-'*30)
print(X_all[numerical_columns].head())
print('='*30)

print(X_all.head())
print('='*30)

################################################################################################
# 1-4. numerical 데이터: skew 줄이기
################################################################################################
from scipy.stats import skew
import numpy as np
print(X_all[numerical_columns].apply(skew).apply(abs).sort_values(ascending =False))
print('-'*30)
X_all[numerical_columns] = X_all[numerical_columns].apply(lambda x: np.log1p(x) if abs(skew(x))>0.2 else x)
print(X_all[numerical_columns].apply(skew).apply(abs).sort_values(ascending =False))
print('='*30)
print(X_all.shape)
print(X_all.isnull().sum())

X_train = X_all[:length_train]
X_test = X_all[length_train:]

################################################################################################
# 2-1. split data
################################################################################################

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)

y_train=y_train["gender"].ravel()
y_val=y_val["gender"].ravel()

print('='*30)
################################################################################################
# 2-2. 모델링: SVC, boost, Kneighbor, RandomForest(v)
################################################################################################

from sklearn.svm import SVC
clf_svc = SVC()
clf_svc.fit(X_train, y_train)
score1 = clf_svc.score(X_train, y_train)
score2 = clf_svc.score(X_val, y_val)
print(score1, score2)
print('-'*30)

from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier()
clf_ada.fit(X_train, y_train)
score1 = clf_ada.score(X_train, y_train)
score2 = clf_ada.score(X_val, y_val)
print(score1, score2)
print('-'*30)

from sklearn.neighbors import KNeighborsClassifier

clf_knc = KNeighborsClassifier(n_neighbors=10)
clf_knc.fit(X_train, y_train)
score1 = clf_knc.score(X_train, y_train)
score2 = clf_knc.score(X_val, y_val)
print(score1, score2)
print('-'*30)

from sklearn.ensemble import RandomForestClassifier

print('='*30)
################################################################################################
# 2-3. GridSearchCV 사용하여 하이퍼파라미터 찾아보기
################################################################################################
from sklearn.model_selection import GridSearchCV
random_search = GridSearchCV(RandomForestClassifier(), param_grid={'min_samples_split': [30,35,40,45], 'max_depth': [5,10,20,30]}, cv=5)
random_search.fit(X_train, y_train)
print(random_search.best_params_)

clf_rf = RandomForestClassifier(min_samples_split=random_search.best_params_['min_samples_split'], max_depth = random_search.best_params_['max_depth'])
print('='*30)

clf_rf.fit(X_train, y_train)
score1 = clf_rf.score(X_train, y_train)
score2 = clf_rf.score(X_val, y_val)
print(score1, score2)
print('-'*30)

print('='*30)
################################################################################################
# 3-1. 평가: 
################################################################################################

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

roc_score = roc_auc_score(y_val, clf_rf.predict_proba(X_val)[:,1])
print(roc_score)
print(confusion_matrix(y_val, clf_rf.predict(X_val)))
print(classification_report(y_val, clf_rf.predict(X_val)))
print('='*30)

################################################################################################
# 평가 결과 대부분 모델에서 validation 모델 스코어가 높지 않음. 62~63%....?
'''
==============================
0.6982857142857143 0.6354285714285715
------------------------------
0.6967619047619048 0.6434285714285715
------------------------------
0.6899047619047619 0.6388571428571429
------------------------------
0.9996190476190476 0.6354285714285715
------------------------------
==============================
0.6601377244842291
[[457  89]
 [230  99]]
              precision    recall  f1-score   support

           0       0.67      0.84      0.74       546
           1       0.53      0.30      0.38       329

    accuracy                           0.64       875
   macro avg       0.60      0.57      0.56       875
weighted avg       0.61      0.64      0.61       875
'''
################################################################################################


################################################################################################
# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
################################################################################################

pred_prob = clf_rf.predict_proba(X_test)[:,1]
ans = pd.DataFrame({'cust_id': X_test_cust_id, 'gender': pred_prob})
ans.to_csv('003000000.csv', index=False)
ret = pd.read_csv("003000000.csv")
print(ret.head())
print('='*30)