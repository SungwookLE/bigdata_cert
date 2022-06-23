import pandas as pd
import numpy as np

import pandas as pd
y_train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/y_train.csv')
X_train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/X_train.csv', encoding='euc-kr')

########################################################################################################################################################################
'''
문제1: 고객의 주 구매지점을 서울 지점과 비서울 지점을 집단으로 구분해 서울 지점과 비 지점간의 고객의 '평균 구매금액'을 대조해 보고자 한다.

이를 위해 주구매지점 중 하단의 8개 지점은 서울 지점, 나머지 지점은 비서울 지점으로 구분하는 변수를 추가한다(서울=1, 비서울=0)
- 서울 8개 지점: 본 점, 잠실점, 강남점, 노원점, 청량리점, 미아점, 관악점, 영등포점
평균구매금액 공식은 다음과 같으며 비서울 지점 대비 서울지점 집단평균구매금액 평균비를 소수 둘째자리까지 구하시오(셋째자리 버림)
'평균구매금액' = '총구매액' / '내점일수'
'''
########################################################################################################################################################################


df = pd.concat([X_train, y_train], axis=1)

seoul = ["본  점", "잠실점", "강남점", "노원점", "청량리점", "미아점", "관악점", "영등포점"]

df["서울"] = df["주구매지점"].apply(lambda x: 1 if x in seoul else 0)
df["평균구매금액"] = df["총구매액"] / df["내점일수"]

print(df.groupby("서울").mean())
print("{:.2f}".format(df.groupby('서울')['평균구매금액'].mean()[1] / df.groupby('서울')['평균구매금액'].mean()[0] ))
print('='*30)


'''
문제2: 문제1에서 추가한 합성변수 '평균구매금액'와 '총구매액', '최대구매액', '환불금액', '내점일수', '주말방문비율', '구매주기' 
7가지 변수를 바탕으로 새로운 데이터셋을 만들고 결측치가 있는 DB는 0으로 채운다.
다음 StandardScaler 방식으로 표준화를 진행한다.
표준화를 진행한 '평균구매금액'과 '총구매액' 변수 둘다 양의 정수의 조건을 충족하는 레코드개수의 수를 구하시오
'''

df2 = df[["평균구매금액", "총구매액", "최대구매액", "환불금액", "내점일수", "주말방문비율", "구매주기"]]
df2["환불금액"].fillna(0, inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df2)
scaled_df2 = pd.DataFrame(data=scaler.transform(df2), columns = df2.columns)

print(scaled_df2.loc[((scaled_df2["평균구매금액"] >= 0) & (scaled_df2["총구매액"] >=0))].shape[0])
print('='*30)

'''
문제3: 문제2에서 진행한 표준화 데이터셋 변수 7가지를 바탕으로 주성분분석(PCA)를 진행한다.(sklearn decomposition PCA 라이브러리 활용)
이때 세번째 주성분까지의 누적 분산 비율을 구하시오 (소수 셋째자리 버림 둘째자리까지 구하시오)
'''

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca.fit(scaled_df2)
pca_scaled_df2 = pd.DataFrame(pca.transform(scaled_df2))

print(np.floor(pca.explained_variance_ratio_.sum()*100)/100)
print('='*30)

'''
문제4: 문제3에서 진행했던 주성분분석 첫번째 주성분과 두번째 주성분의 아래 공식을 대입한 값을 소수 둘째자리까지 구하시오(셋째자리 버림)
'''

print(pca_scaled_df2)
print(np.sqrt(np.sum((pca_scaled_df2[0] - pca_scaled_df2[1])**2)))
print('='*30)


'''
문제5: 문제3에서 진행했던 주성분분석 첫번째부터 네번째 주성분 데이터셋을 바탕으로 군집분석을 수행한다. 
방법은 계층적 군집분석이고 계산법은 유클리드 거리, 와드연결법으로 수행한다. 군집을 3개로 나눈다고 할때 레코드 수가 가장 많은 군집의 레코드수를 구하여라
'''


from sklearn.cluster import AgglomerativeClustering

cluser = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage='ward')
cluser_ret=cluser.fit_predict(pca_scaled_df2)
print(pd.Series(cluser_ret).value_counts().sort_values(ascending=False)[0])

from sklearn.metrics import silhouette_score
score = silhouette_score(pca_scaled_df2, cluser_ret)
print(score)

print(cluser.labels_.shape)