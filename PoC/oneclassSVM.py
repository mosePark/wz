'''
25-07-10
oneclass SVM
'''

# 전체 코드: One-Class SVM 기반 이상탐지

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1) 데이터 로드 & 전처리
link = "C:/Users/BEGAS-NB-151/Desktop/begas/프로젝트/wezon/"
df = pd.read_excel(link + 'Raw_데이터.xlsx')
df['parent_tag'] = df['tag_code'].str.split('.').str[:2].str.join('.')
df['data_date'] = pd.to_datetime(df['data_date'])

# 2) 데이터 준비 함수
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

# 3) One-Class SVM 이상탐지 함수
def ocsvm_detect(X, train_frac=0.7, nu=0.01, kernel='rbf', gamma='scale'):
    n_train = int(len(X) * train_frac)
    scaler = StandardScaler().fit(X[:n_train])
    X_s = scaler.transform(X)
    
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_s[:n_train])
    
    scores = model.decision_function(X_s)
    preds  = model.predict(X_s)
    # preds: +1 정상, -1 이상
    return scores, preds, n_train

# 4) 실행 & 시각화
first_tag = df['parent_tag'].unique()[0]
pivot = prepare_data(df, first_tag)
X = pivot.values

scores, preds, n_train = ocsvm_detect(X, train_frac=0.9)
time_idx = pivot.index

plt.figure(figsize=(10,4))
plt.plot(time_idx, scores, label='Decision function')
plt.axhline(0, color='red', linestyle='--', label='Threshold = 0')
# 이상치 마커
anomalies = time_idx[preds == -1]
plt.scatter(anomalies, np.zeros_like(anomalies), color='red', s=50, marker='x', label='Anomaly')

plt.title(f'One-Class SVM 이상탐지: {first_tag}')
plt.xlabel('Time')
plt.ylabel('Score')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
