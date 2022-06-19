# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

'''
#####################################################################################
작성일: '22.6/15
작성자: 이성욱
샘플 문제: https://dataq.goorm.io/exam/116674/%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/3
#####################################################################################
'''

# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# 사용자 코딩

## 1. 데이터 분석(EDA)
### 칼럼 타입 정보
print(f"X_train shape is {X_train.shape} and columes are \n {X_train.columns}")
print(f"X_test shape is {X_test.shape}. and columes are \n {X_test.columns}")
print(f"y_train shape is {y_train.shape}. and columes are \n {y_train.columns}")
print(X_train.dtypes)
print(f"NULL information is \n {X_train.isnull().sum()}")
### 결측값 채우기
X_train["환불금액"].fillna(0, inplace=True)
### 산술정보
extract_int_df = X_train.select_dtypes(exclude='object').drop('cust_id', axis=1)
print(extract_int_df.describe())
### 상관관계 분석
df = pd.concat([extract_int_df, y_train.drop('cust_id',axis=1)], axis=1)
corrmat=df.corr()
result = abs(corrmat['gender']).sort_values(ascending=False)
print(result)

## 2. Feature Engineering
### 1. Categorical / Numerical 분리
X_train_without_id = X_train.drop('cust_id',axis=1)
X_train_categorical = X_train_without_id.select_dtypes(include='object')
X_train_numerical = X_train_without_id.select_dtypes(exclude='object')

print(f"Categorical columns are \n{X_train_categorical.columns}.")
print(f"Numerical columns are \n{X_train_numerical.columns}.")

### 2. categorical: one-hot encoding
X_train_categorical=pd.get_dummies(X_train_categorical)
print(X_train_categorical.columns)

### 3. numerical: skewness
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_numerical)
X_train_numerical_scaled = pd.DataFrame(scaler.transform(X_train_numerical), columns = X_train_numerical.columns)

from scipy.stats import norm, skew # for some statistics
skewness_before = X_train_numerical_scaled.apply(lambda x : skew(x))
skewness_features = skewness_before[abs(skewness_before.values)>1].sort_values(ascending=False).index
print(f"Before: \n {skewness_before[skewness_features]}")

import numpy as np
for col in skewness_features:
	X_train_numerical_scaled[col] = np.log1p(X_train_numerical_scaled[col])
skewness_after = X_train_numerical_scaled.apply(lambda x : skew(x))

print(f"After: \n {skewness_after[skewness_features]}")

### 전처리 분석 끝난 데이터 다시 합쳐주기

X_train_after = pd.concat([X_train_categorical, X_train_numerical_scaled], axis=1)
y_train_gender = y_train['gender'].ravel()

## 3. 모델링
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_after, y_train_gender)
pred_train = clf.predict(X_train_after)
train_score = clf.score(X_train_after, y_train_gender)
print(f"Train Score is {train_score}")

## 4. 분석 및 시각화 (ROC-AUC), 테스트결과 (to_csv())
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train_gender, pred_train)
print(f"훈련데이터 오차행렬: \n {confusion_train}")

from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train_gender, pred_train)
print(f"분류예측 레포트: \n {cfreport_train}")





### 끝

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)
