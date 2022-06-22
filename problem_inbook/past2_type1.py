#####################################################
'''
1-1. 다음은 Boston Housing 데이터셋이다. crim 항목의 상위에서 10번째 값
(즉, 상위 10번째 값 중에서 가장 작은 값)으로 상위 10개의 값을 변환하고
age 80 이상인 값에 대하여 crim 평균을 구하시오
'''
#####################################################
import pandas as pd
import numpy as np

data = np.array([[90,100,10,20,40,60], [90,100,10,20,40,60]]).transpose()
print(data.shape)
df = pd.DataFrame(data=data ,columns = ["crim", "age"])

df2 = df.sort_values(by='crim', ascending=True).head(10).reset_index(drop=True)
print(df2)
mean = df2.loc[df2["age"]>=80, "crim"].mean()
print(mean)

#####################################################
'''
1-2. 주어진 데이터의 첫번째 행부터 순서대로 80% 까지의 데이터를 훈련 데이터로 추출 후
'total_bedrooms' 변수의 결측값(NA)을 'total_bedrooms' 변수의 중앙값으로 대체하고
대체 전의 'total_bedrooms' 변수 표준편차 값의 차이의 절대값을 구하시오.
'''
#####################################################

data = np.array([[np.NaN,100,10,np.NaN,40,60], [90,100,10,20,40,60], [90,100,10,20,40,60]]).transpose()
df = pd.DataFrame(data=data ,columns = ["total_bedrooms", "age", "room"])

before =  df["total_bedrooms"].std(ddof=1) #degree of freedom is defaultly 1. ==> 1 means s^2 = \Sigma[(x-u)^2]/(N-1) / 0 means Sigma^2 = \Sigma[(x-u)^2]/(N)
print(before)
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
after =  df["total_bedrooms"].std()
print(after)
print(abs(after-before))

#####################################################
'''
1-3. 다음은 insurance 데이터셋이다. charges 항목의 이상값의 합을 구하시오.
(이상값이 평균에서 1.5 표준편차 이상인 값)
'''
#####################################################

data = np.array([[np.NaN,100,10,np.NaN,40,60], [90,100,10,20,40,200], [90,100,10,20,40,60]]).transpose()
df = pd.DataFrame(data=data ,columns = ["age", "charges", "room"])

u = df["charges"].mean()
s = df["charges"].std()

print(u, s)
outlier = df.loc[ abs(df["charges"] - u) > 1.5*s, "charges" ]
print(outlier.sum())