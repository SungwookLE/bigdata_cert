##################################################################################################
## 작업 2유형
'''
 - 타이타닉 생존자 예측
 - 데이터 출처 : https://www.kaggle.com/competitions/titanic/overview
'''
##################################################################################################

from subprocess import check_output
import pandas as pd
import numpy as np

print(check_output(["ls", "data_manim/kaggle_titanic"]).decode('utf8'))
train_df = pd.read_csv("data_manim/kaggle_titanic/train.csv")
test_df = pd.read_csv("data_manim/kaggle_titanic/test.csv")

print(train_df.head(3))
print(test_df.head(3))

x_train = train_df.drop(["Survived", "PassengerId"], axis = 1)
y_train = train_df[["Survived"]]
x_test = (test_df.copy()).drop("PassengerId", axis=1)

##################################################################################################
'''
## 1. 데이터 분석(EDA)
- Exploratory Data Analysis
- 데이터 차원, 형태 파악하기
- 그래프 그려서 예측변수와 다른 변수와의 상관관계 파악하기
'''
##################################################################################################
print("="*50)
print("##1. 데이터 분석(EDA)")

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")

corr= train_df.corr()
print(f"corr: {corr.apply(abs).sort_values(by='Survived', ascending=False)['Survived']}")

print(x_train.dtypes)

##################################################################################################
'''
## 2. Feature Engineering
- **Tip**: kaggle 등의 정리된 데이터를 풀 때에는 train, test `pd.concat` 해서 결측치 처리 등을 한번에 처리하고 분리해주는게 편리하다.

### 2-1. categorical, numerical features 분리
- using `select_dtypes()`
- numerical 데이터 중에서도 month나 year등의 데이터는 categorical로 분류해주기 `apply(str)`

### 2-2. 비어있는 missing 결측값 채우기
- numerical: mean, median, mode 등을 활용하여 데이터 채우기 `.fillna(), mean(), median(), mode()`

### 2-3. categorical 데이터의 one-hot 인코딩
- catergorical: `pd.get_dummies()` 또는 `LabelEncoder`를 활용하여 missing 데이터 없애고, one-hot encoding 해주기

### 2-4. numerical 데이터 skewness 줄이기
- numerical data의 skewness줄이기
- `from scipy.stats import skew`

### 2-5. outlier 값 제거하기
- *train dataset*을 대상으로만 제거할것
```
for col in x_numerical_columns:
    col_IQR = x_train[col].quantile(0.75) - x_train[col].quantile(0.25)
    x_train = x_train.loc[x_train[col].between(x_train[col].quantile(
        q=0.25)-1.5*col_IQR, x_train[col].quantile(q=0.75)+1.5*col_IQR)]

y_train = y_train.loc[x_train.index]
```
- 제거하는 방법 말고, 아웃라이어를 min-max bound 처리해주는 방법도 있는데, 일종의 전처리함수처럼 사용해도됨
    - 전처리함수처럼 사용한다는 의미는 test data도 전처리할때 해주어야한다는 말
    - 물론 안해줬을 때, 성능이 더 잘나온다면, 성능을 기준으로 결정해야할 문제

### 2-6. new features / del features
- 필요하다면
'''
##################################################################################################
print("="*50)

len_x_train = len(x_train)
len_x_test = len(x_test)

x_all = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)

print("2-1. 결측값 처리")
print("Before: ", x_all.isnull().sum())

x_all["Age"].fillna(x_all["Age"].dropna().mean(), inplace=True)
x_all["Fare"].interpolate(method="values", inplace=True)
x_all.drop("Cabin", axis=1, inplace=True)
x_all["Embarked"].fillna(x_all["Embarked"].mode()[0], inplace=True)
print("After: ", x_all.isnull().sum())

print("2-2. 카테고리칼 뉴메리칼 분리")
x_numerical_features = x_all.select_dtypes(exclude="object")
x_categorical_features = x_all.select_dtypes(include="object")

print("2-3. categorical one-hot")
x_categorical_onehot = pd.get_dummies(x_categorical_features)

print("2-4. numerical skewness for over the 0.5")
from scipy.stats import skew
print("Before Skew: \n", x_numerical_features.apply(skew).apply(abs).sort_values(ascending=False))
x_numerical_features=x_numerical_features.apply(lambda x: np.log1p(x) if skew(x) >= 0.5 else x)
print("After Skew: \n",x_numerical_features.apply(skew).apply(abs).sort_values(ascending=False))


x_all = pd.concat([x_numerical_features, x_categorical_onehot], axis=1)
x_train = x_all[:len_x_train]
x_test = x_all[len_x_train:]

print("2-5. outlier 확인하기")
# from plot_box_for_outlier_quantile import plot_box_per_column
# plot_box_per_column(x_train, x_numerical_features.columns, ratio_IQR=1.5)

print(f"Before outlier: \n {x_train.shape}")
for col in x_numerical_features.columns:
    col_IQR = x_train[col].quantile(0.75) - x_train[col].quantile(0.25) 
    x_train = x_train.loc[x_train[col].between(x_train[col].quantile(0.25)-1.5*col_IQR, x_train[col].quantile(0.75)+1.5*col_IQR)]
    y_train = y_train.loc[x_train.index]

print(f"After outlier: \n {x_train.shape}")

##################################################################################################
'''
## 3. 모델링
- Dataset: `train_test_split`
- CrossValidation using `cross_val_score, KFold`
- Preprocessing: `StandardScaler, RobustScaler`
- [Regressor](https://scikit-learn.org/stable/search.html?q=Regress): `LinearRegression, RidgeCV, LassoCV, ElasticNetCV...`
- [Classifier](https://scikit-learn.org/stable/search.html?q=classifier): `KNeighborsClassifier, RandomForestClassifier, ...`
- Easy modeling: `make_pipeline`

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)
kNN = make_pipeline(RobustScaler(),KNeighborsClassifier(n_neighbors=13))
score = cross_val_score(kNN, train_data, y_label, cv= k_fold, n_jobs =1 , scoring='accuracy')
print(np.mean(score))
'''
##################################################################################################

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

k_fold = KFold(n_splits=5, shuffle=True)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.3)

clf_rf = make_pipeline(MinMaxScaler(), RandomForestClassifier())
#clf_rf.fit(x_train, y_train)
score_rf = cross_val_score(clf_rf, x_train, y_train, cv= k_fold, n_jobs =1 , scoring='accuracy')

clf_lr = make_pipeline(MinMaxScaler(), LogisticRegression())
#clf_lr.fit(x_train, y_train) cross_val_score를 쓰면 할 필요가 없음
score_lr = cross_val_score(clf_lr, x_train, y_train, cv= k_fold, n_jobs =1 , scoring='accuracy')

clf_knc = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
#clf_lr.fit(x_train, y_train) cross_val_score를 쓰면 할 필요가 없음
score_knc = cross_val_score(clf_knc, x_train, y_train, cv= k_fold, n_jobs =1 , scoring='accuracy')


clf_svc = make_pipeline(MinMaxScaler(),SVC())
score_svc = cross_val_score(clf_svc, x_train, y_train, cv= k_fold, n_jobs =1 ,  scoring='accuracy')

print(f"1-1. rf cross_val score is {np.mean(score_rf)}")
print(f"2-1. lr cross_val score is {np.mean(score_lr)}")
print(f"3-1. knc cross_val score is {np.mean(score_knc)}")
print(f"4-1. svc cross_val score is {np.mean(score_svc)}")



