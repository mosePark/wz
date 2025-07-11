'''
참고 : https://github.com/iqiukp/SVDD-Python/tree/master
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import BaseSVDD

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1) 데이터 로드 & 전처리
link = "C:/Users/BEGAS-NB-151/Desktop/begas/프로젝트/wezon/"
df = pd.read_excel(link + 'Raw_데이터.xlsx')
df['parent_tag'] = df['tag_code'].str.split('.').str[:2].str.join('.')
df['data_date'] = pd.to_datetime(df['data_date'])

# 2) prepare_data 함수 (기존)
def prepare_data(df, tag, freq='5S'):
    grp = df[df['parent_tag'] == tag]
    pivot = (grp
             .pivot_table(index='data_date',
                          columns='tag_code',
                          values='data_value',
                          aggfunc='mean')
             .sort_index()
             .resample(freq).asfreq()
             .dropna())
    return pivot

# 3) 분석할 태그를 수동 지정
target_tag = df['parent_tag'].unique()[0]   # ← 여기 원하는 parent_tag로 바꿔주세요
freq       = '5S'        # 필요에 따라 조정

# 4) 해당 태그 데이터만 가져오기
data = prepare_data(df, target_tag, freq=freq)
X = data.values

# 5) SVDD 모델 초기화 & 학습
svdd = BaseSVDD.BaseSVDD(
    C=0.9,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=1.0,
    display='off',
    n_jobs=-1
)
svdd.fit(X)

# 6) 이상치 예측
scores    = svdd.decision_function(X).ravel()
labels    = svdd.predict(X).ravel()
anomalies = np.where(labels == -1)[0]

# 7) 시각화 (한 플롯으로 간단히)
plt.figure(figsize=(12,5))
plt.plot(data.index, scores, label='SVDD Score')
plt.scatter(data.index[anomalies],
            scores[anomalies],
            color='red', marker='x', s=80,
            label='Anomaly')
plt.title(f"SVDD 이상탐지 ({target_tag})")
plt.xlabel("시간")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()
