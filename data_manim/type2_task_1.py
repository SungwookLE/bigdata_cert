##################################################################################################
## 작업 2유형
'''
 - 서비스 이탈예측 데이터
 - 데이터 설명 : 고객의 신상정보 데이터를 통한 회사 서비스 이탈 예측 (종속변수 : Exited)
 x_train : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv
 y_train : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv
 x_test : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv
 x_label(평가용) : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_test.csv 
 데이터 출처 : https://www.kaggle.com/shubh0799/churn-modelling 에서 변형
'''
##################################################################################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, scale
import numpy as np
from scipy.stats import skew
import pandas as pd

from plot_box_for_outlier_quantile import plot_box_per_column

#데이터 로드
x_train = pd.read_csv(
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv")
y_train = pd.read_csv(
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv")
x_test = pd.read_csv(
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv")

print(x_train.head())
print(y_train.head())

print('='*50, "예측변수: Exited", '='*50)

##################################################################################################
'''
1. 데이터분석(EDA)
- 데이터 차원, 형태 파악하기
- 상관관계 파악하기
'''
print('='*50)
##################################################################################################

train_merge = pd.merge(x_train, y_train, on="CustomerId")
print("1-1. 칼럼별 데이터 타입 보기, shape")
print(train_merge.dtypes)
print(train_merge.shape)

print("1-2. 상관관계 살펴보기")
corr = train_merge.drop("CustomerId", axis=1).corr()
print(corr["Exited"].apply(abs).sort_values(ascending=False))

y_train = train_merge[["Exited"]]
x_train = train_merge.drop("Exited", axis=1)
len_train = len(x_train)
len_test = len(x_test)

print("1-3. numerical columns의 산술정보")
print(x_train.describe())

##################################################################################################
'''
2. Feature Engineering
- Tip: train, test `concat` 해서 한번에 가공해주는게 편리함
2-1. categorical, numerical 분리
2-2. 비어있는 missing 데이터 채우기
2-3. categorical의 one-hot encoding 줄이기
2-4. numerical의 skewness 줄이기, outlier 없애기
2-5. new / del features
'''
print('='*50)
##################################################################################################

print("2-1. Feature Engineering을 위해 데이터 합치고 타입별 분리하기")
x_all = pd.concat([x_train, x_test], axis=0).drop("CustomerId", axis=1)

x_numerical = x_all.select_dtypes(exclude='object')
x_categorical = x_all.select_dtypes(include='object')

x_numerical_columns = x_numerical.columns
x_categorical_columns = x_categorical.columns

print("2-2. missing data 처리: 없음")
print(x_numerical.isnull().sum())
print(x_categorical.isnull().sum())

print("2-3. categorical: one-hot encoding 처리")
x_categorical_onehot = pd.get_dummies(x_categorical)
x_categorical_onehot_columns = x_categorical_onehot.columns
print(x_categorical_onehot.head(3))

print("2-4. numerical: skewness 처리하기")
skew_over = np.abs(skew(x_numerical)) > 0.5
skew_columns = x_numerical_columns[skew_over]
print(f"before skewness: {np.abs(skew(x_numerical))}, {x_numerical.columns}")

skew_after = pd.DataFrame(data=np.log1p(
    x_numerical[skew_columns]), columns=skew_columns)
x_numerical = x_numerical.drop(skew_columns, axis=1)
x_numerical = pd.concat([x_numerical, skew_after], axis=1)
print(f"after skewness: {np.abs(skew(x_numerical))}, {x_numerical.columns}")

##################################################################################################
# x_all 에서 train / test 데이터 분리
##################################################################################################

x_all = pd.concat([x_numerical, x_categorical_onehot], axis=1)
x_train = x_all[:len_train]
x_test = x_all[len_train:]


print("2-5. outlier 처리하기(IQR 기준으로) only in traindata")
for col in x_numerical_columns:
    col_IQR = x_train[col].quantile(0.75) - x_train[col].quantile(0.25)
    x_train = x_train.loc[x_train[col].between(x_train[col].quantile(
        q=0.25)-1.5*col_IQR, x_train[col].quantile(q=0.75)+1.5*col_IQR)]

y_train = y_train.loc[x_train.index]
print(
    f" x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}")

x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)


##################################################################################################
'''
3. 모델링
'''
print('='*50)
##################################################################################################


scaler = MinMaxScaler()
scaler.fit(x_train[x_numerical_columns])
x_train_scaled = scaler.transform(x_train[x_numerical_columns])
x_test_scaled = scaler.transform(x_test[x_numerical_columns])

x_train_scaled = pd.concat([x_train[x_categorical_onehot_columns], pd.DataFrame(
    data=x_train_scaled, columns=x_numerical_columns)], axis=1)
x_test_scaled = pd.concat([x_test[x_categorical_onehot_columns], pd.DataFrame(
    data=x_test_scaled, columns=x_numerical_columns)], axis=1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_scaled, y_train, stratify=y_train, test_size=0.3)

print("3-1. train_test_split")

print(type(y_train))
print(y_train)

print(f"x_train {x_train.shape}, x_val {x_val.shape}, y_train {y_train.shape}, y_val {y_val.shape}")
print(y_train.head())

##################################################################################################
# pd.get_dummies(y_train["Exited"]) 처리해서 one-hot 인코딩 해주거나,
# astype(int)로 처리해도 됨, 단 y_label이 string인 경우에는 one-hot incoding.
# ex: y_train = pd.get_dummies(y_train)
##################################################################################################

print("3-2. RandomForestClassifier 분류")

print("3-2-1. Hyperparameter Random Search")
param_distibs = {"n_estimators": np.random.randint(20, 500, size=1)}
random_search = RandomizedSearchCV(RandomForestClassifier(
), param_distributions=param_distibs, cv=5, return_train_score=True)
random_search.fit(x_train, y_train)

print(f"Best Parameter is {random_search.best_params_}")
print(f"Best CV Score is {random_search.best_score_.round(3)}")

print("3-2-2. RandomForestClassifier 분류")
clf_rf = RandomForestClassifier(random_search.best_params_["n_estimators"])
clf_rf.fit(x_train, y_train)
arr = np.array([clf_rf.feature_importances_,
               x_train_scaled.columns]).transpose()
feature_importance_df = pd.DataFrame(data=arr, columns=["var", "col"])
print(
    f"Important Features are {feature_importance_df.sort_values(by='var', ascending=False)}")

print(f"train score is {clf_rf.score(x_train, y_train)}")
print(f"validation score is {clf_rf.score(x_val, y_val)}")
y_rf_val_pred = clf_rf.predict(x_val)
print(f"test predict is {clf_rf.predict(x_test)}")
#print(f"test proba is {clf_rf.predict_proba(x_test)}")


print("3-3. Multi-layer Perceptron classifier 분류")

clf_mlp = MLPClassifier()
clf_mlp.fit(x_train, y_train)

print(f"train score is {clf_mlp.score(x_train, y_train)}")
print(f"validation score is {clf_mlp.score(x_val, y_val)}")
y_mlp_val_pred = clf_mlp.predict(x_val)
print(f"test predict is {clf_mlp.predict(x_test)}")
#print(f"test proba is {clf_mlp.predict_proba(x_test)}")

##################################################################################################
'''
4. 평가, 분석 및 시각화
- Confusion matrix, ROC curve
'''
##################################################################################################

print('='*50)

##################################################################################################
# sklearn 에서의 confusion mtx 출력값
#  [[TN FP]
#  [FN TP]]
##################################################################################################

print(f"randomforest의 confusion mtx: {confusion_matrix(y_val, y_rf_val_pred)}")
print(classification_report(y_val, y_rf_val_pred))
roc_auc = roc_auc_score(y_val, clf_rf.predict_proba(x_val)[:, 1])
print(roc_auc)

# print('-'*50)
# print(clf_rf.predict_proba(x_val)[:, 0])
# print(clf_rf.predict_proba(x_val)[:, 1])
# print(clf_rf.predict(x_val))
# print('-'*50)

print(f"mlp의 confusion mtx: {confusion_matrix(y_val, y_mlp_val_pred)}")
print(classification_report(y_val, y_mlp_val_pred))
roc_auc = roc_auc_score(y_val, clf_mlp.predict_proba(x_val)[:, 1])
print(roc_auc)

##################################################################################################
## 끝
##################################################################################################
