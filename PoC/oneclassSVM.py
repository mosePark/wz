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
def ocsvm_detect(X, train_frac=0.95, nu=0.1, kernel='rbf', gamma='scale'):
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

scores, preds, n_train = ocsvm_detect(X, train_frac=0.95)
time_idx = pivot.index

# 수동 임계값 설정 (더 엄격하게)
threshold = -0.05  # 또는 np.percentile(scores, 5)
manual_preds = np.where(scores < threshold, -1, 1)

# 시각화
plt.figure(figsize=(10,4))
plt.plot(time_idx, scores, label='Decision function')
plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
# 이상치 마커 (수동 임계값 사용)
anomalies = time_idx[manual_preds == -1]
plt.scatter(anomalies, np.full_like(anomalies, threshold, dtype=float), 
           color='red', s=50, marker='x', label='Anomaly')

plt.title(f'One-Class SVM 이상탐지: {first_tag}')
plt.xlabel('Time')
plt.ylabel('Score')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"탐지된 이상치 개수: {np.sum(manual_preds == -1)}")
print(f"전체 데이터 대비 이상치 비율: {np.sum(manual_preds == -1)/len(manual_preds):.2%}")

# 점수 분포 확인
plt.figure(figsize=(10,3))
plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel('Decision function score')
plt.ylabel('Frequency')
plt.title('Decision Function Score 분포')
plt.legend()
plt.tight_layout()
plt.show()
