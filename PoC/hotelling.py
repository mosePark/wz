# %% 
# 1) 라이브러리 로드
import numpy as np
import pandas as pd
import openpyxl

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# %% 
# 1) 원본 데이터 로드 & 전처리
link = "C:/Users/BEGAS-NB-151/Desktop/begas/프로젝트/wezon/"
df = pd.read_excel(link + 'Raw_데이터.xlsx')

df['parent_tag'] = df['tag_code'].str.split('.').str[:2].str.join('.')
df['data_date'] = pd.to_datetime(df['data_date'])

# %%

df['parent_tag'].value_counts()

# %%

df.loc[df['parent_tag'] == 'hpc1.uh', :]

# %%

# hpc1.uh 그룹 원본 행 수 vs. 고유 타임스탬프 개수
grp = df[df['parent_tag']=='hpc1.uh']

total_rows      = len(grp)                   # 1908
unique_timestamps = grp['data_date'].nunique()  # 53

grp['data_date'].unique()

df['data_date'].unique()

print(f"원본 행 수: {total_rows}")
print(f"고유 data_date 개수: {unique_timestamps}")


# %%

# 2) T² 통계량
def calc_t2(X, train_frac=.9):
    n, p = X.shape
    n_train = int(n * train_frac)
    X_train = X[:n_train]
    mu = X_train.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(X_train, rowvar=False))
    diffs = X - mu
    t2 = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    return t2, n_train, p

# 3) 그룹별 이상탐지
results = {}
for tag, grp in df.groupby('parent_tag'):
    # a) wide-format
    pivot = grp.pivot_table(
        index='data_date',
        columns='tag_code',
        values='data_value',
        aggfunc='mean'
    )
    pivot = pivot.sort_index().resample('5S').asfreq().dropna()
    
    # b) T² 계산
    X = pivot.values
    t2, n_train, p = calc_t2(X)
    
    # c) 임계치(χ² 95%)
    chi95 = stats.chi2.ppf(0.95, df=p)
    outliers = pivot.index[t2 > chi95]
    
    # d) 결과 저장
    results[tag] = {
        'time_index': pivot.index,
        't2_scores': t2,
        'threshold': chi95,
        'outliers': outliers
    }

# %%
# 4) 결과 확인 & 시각화 (예: 첫 번째 그룹)
first_tag = list(results)[10]
res = results[first_tag]

plt.figure(figsize=(10,4))
plt.scatter(res['time_index'], res['t2_scores'], s=10)
plt.axhline(res['threshold'], color='red', linestyle='--')
plt.title(f'Hotelling T² for {first_tag}')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"[{first_tag}] 총 {len(res['t2_scores'])}개 시점 중 이상치 {len(res['outliers'])}개:")
for t in res['outliers']:
    print("  ", t)

# %% 
# 5) ARL0 및 FTS 계산을 위한 synthetic shift 적용 (mu 추가)
metrics = []

for tag, grp in df.groupby('parent_tag'):
    # a) pivot & resample & dropna
    pivot = (
        grp
        .pivot_table(
            index='data_date',
            columns='tag_code',
            values='data_value',
            aggfunc='mean'
        )
        .sort_index()
        .resample('5S').asfreq()
        .dropna()
    )
    
    # b) 원본 X 및 학습 구간 설정
    X = pivot.values
    t2, n_train, p = calc_t2(X)
    threshold     = stats.chi2.ppf(0.95, df=p)
    
    # c) 학습 구간 평균(mu)과 표준편차(sigma) 계산
    X_train     = X[:n_train]
    mu          = X_train.mean(axis=0)
    sigma_train = X_train.std(axis=0)
    
    # d) ARL0 계산 (in-control 구간 오탐까지 포인트 수)
    train_t2 = t2[:n_train]
    fa_idx   = np.where(train_t2 > threshold)[0]
    ARL0     = fa_idx[0] + 1 if fa_idx.size else n_train
    
    # e) synthetic shift: 테스트 첫 포인트부터 5σ 이동
    X_synth     = X.copy()
    shift_start = n_train + 1
    X_synth[shift_start:, :] = mu + sigma_train * 3
    
    # f) 다시 T² 계산
    t2_synth, _, _ = calc_t2(X_synth)
    test_t2_synth = t2_synth[n_train:]
    
    # g) FTS 계산 (shift 이후 첫 신호까지 포인트 수)
    post_shift = test_t2_synth[1:]  # shift_start 바로 다음부터
    sig_idx    = np.where(post_shift > threshold)[0]
    FTS        = sig_idx[0] + 1 if sig_idx.size else np.nan
    
    metrics.append({
        'parent_tag':    tag,
        'ARL0':          ARL0,
        'FTS (points)':  FTS
    })

# %%

# 6) 결과 요약
metrics_df = pd.DataFrame(metrics).set_index('parent_tag')
print(metrics_df)

# %% #########################################################

# # %%
# # 1) 데이터 품질 체크 추가
# def check_data_quality(df):
#     """데이터 품질 검증"""
#     print("=== 데이터 품질 체크 ===")
#     print(f"전체 행 수: {len(df)}")
#     print(f"결측치: {df.isnull().sum().sum()}")
#     print(f"중복 행: {df.duplicated().sum()}")
    
#     # 각 parent_tag별 시간 간격 체크
#     for tag in df['parent_tag'].unique():
#         grp = df[df['parent_tag'] == tag]
#         time_diffs = grp['data_date'].sort_values().diff().dropna()
#         print(f"\n[{tag}] 평균 시간 간격: {time_diffs.mean()}")

# check_data_quality(df)

# # %%
# # 2) T² 계산 함수 개선
# def calc_t2_robust(X, train_frac=0.9, min_samples=30):
#     """개선된 T² 계산 (검증 추가)"""
#     n, p = X.shape
    
#     # 최소 샘플 수 체크
#     if n < min_samples:
#         print(f"경고: 샘플 수({n})가 너무 적습니다. 최소 {min_samples}개 필요")
#         return None, None, None
    
#     # 샘플 수 > 변수 수 체크
#     if n <= p:
#         print(f"경고: 샘플 수({n}) <= 변수 수({p}). 공분산 행렬 계산 불가")
#         return None, None, None
    
#     n_train = int(n * train_frac)
#     X_train = X[:n_train]
    
#     # 표준화 (스케일 차이 보정)
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_scaled = scaler.transform(X)
    
#     mu = np.zeros(p)  # 표준화 후 평균은 0
#     cov = np.cov(X_train_scaled, rowvar=False)
    
#     # 조건수 체크 (다중공선성)
#     cond_num = np.linalg.cond(cov)
#     if cond_num > 1000:
#         print(f"경고: 조건수가 높음 ({cond_num:.1f}). 다중공선성 가능성")
    
#     cov_inv = np.linalg.pinv(cov)
#     diffs = X_scaled - mu
#     t2 = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    
#     return t2, n_train, p

# # %%
# # 3) 임계값 설정 개선
# def get_threshold(t2_train, p, method='percentile', alpha=0.05):
#     """다양한 임계값 설정 방법"""
#     if method == 'chi2':
#         # 카이제곱 분포 (정규성 가정)
#         return stats.chi2.ppf(1-alpha, df=p)
    
#     elif method == 'percentile':
#         # 경험적 백분위수 (정규성 가정 불필요)
#         return np.percentile(t2_train, (1-alpha)*100)
    
#     elif method == 'beta':
#         # F분포 기반 (작은 샘플에 적합)
#         n = len(t2_train)
#         f_critical = stats.f.ppf(1-alpha, p, n-p)
#         return p*(n-1)/(n-p) * f_critical

# # %%
# # 4) 개선된 이상탐지 루프
# results_improved = {}

# for tag, grp in df.groupby('parent_tag'):
#     print(f"\n처리 중: {tag}")
    
#     # 피벗 테이블
#     pivot = grp.pivot_table(
#         index='data_date',
#         columns='tag_code',
#         values='data_value',
#         aggfunc='mean'
#     )
    
#     # 리샘플링 전후 데이터 손실 체크
#     before_resample = len(pivot)
#     pivot = pivot.sort_index().resample('5S').asfreq().dropna()
#     after_resample = len(pivot)
    
#     if after_resample < before_resample * 0.5:
#         print(f"  경고: 리샘플링으로 {before_resample-after_resample}개 행 손실")
    
#     # T² 계산
#     X = pivot.values
#     t2, n_train, p = calc_t2_robust(X)
    
#     if t2 is None:
#         print(f"  {tag} 건너뜀")
#         continue
    
#     # 여러 임계값 비교
#     chi95 = get_threshold(t2[:n_train], p, method='chi2')
#     pct95 = get_threshold(t2[:n_train], p, method='percentile')
    
#     print(f"  임계값 - Chi2: {chi95:.2f}, Percentile: {pct95:.2f}")
    
#     # 이상치 탐지
#     outliers = pivot.index[t2 > chi95]
    
#     results_improved[tag] = {
#         'time_index': pivot.index,
#         't2_scores': t2,
#         'threshold_chi2': chi95,
#         'threshold_pct': pct95,
#         'outliers': outliers,
#         'n_train': n_train,
#         'n_features': p
#     }

# # %%
# # 5) FTS 계산 개선 (shift 크기별)
# def calculate_metrics_by_shift(X, n_train, p, shift_sizes=[1, 3, 5]):
#     """다양한 shift 크기에 대한 FTS 계산"""
#     metrics = {}
    
#     for shift_sigma in shift_sizes:
#         X_synth = X.copy()
#         mu = X[:n_train].mean(axis=0)
#         sigma = X[:n_train].std(axis=0)
        
#         # Shift 적용
#         X_synth[n_train:, :] = mu + sigma * shift_sigma
        
#         # T² 재계산
#         t2_synth, _, _ = calc_t2(X_synth, train_frac=n_train/len(X))
#         threshold = stats.chi2.ppf(0.95, df=p)
        
#         # FTS 계산
#         test_t2 = t2_synth[n_train:]
#         sig_idx = np.where(test_t2 > threshold)[0]
#         FTS = sig_idx[0] + 1 if sig_idx.size else len(test_t2)
        
#         metrics[f'FTS_{shift_sigma}σ'] = FTS
    
#     return metrics

# # %%
# # 6) 결과 시각화 개선
# def plot_t2_with_details(tag, result, figsize=(12, 6)):
#     """상세 정보가 포함된 T² 플롯"""
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
#                                    gridspec_kw={'height_ratios': [3, 1]})
    
#     # T² 스코어 플롯
#     time_idx = result['time_index']
#     t2_scores = result['t2_scores']
#     n_train = result['n_train']
    
#     # 학습/테스트 구간 구분
#     ax1.scatter(time_idx[:n_train], t2_scores[:n_train], 
#                 s=10, alpha=0.6, label='Train', color='blue')
#     ax1.scatter(time_idx[n_train:], t2_scores[n_train:], 
#                 s=10, alpha=0.6, label='Test', color='green')
    
#     # 임계값들
#     ax1.axhline(result['threshold_chi2'], color='red', 
#                 linestyle='--', label=f'Chi2 (95%): {result["threshold_chi2"]:.2f}')
#     ax1.axhline(result['threshold_pct'], color='orange', 
#                 linestyle=':', label=f'Percentile (95%): {result["threshold_pct"]:.2f}')
    
#     # 이상치 표시
#     outlier_idx = result['outliers']
#     outlier_scores = t2_scores[time_idx.isin(outlier_idx)]
#     ax1.scatter(outlier_idx, outlier_scores, 
#                 color='red', s=50, marker='x', label='Outliers')
    
#     ax1.set_ylabel('T² Score')
#     ax1.set_title(f'Hotelling T² for {tag} (n_features={result["n_features"]})')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # 기여도 플롯 (어떤 변수가 이상치에 기여했는지)
#     # 여기서는 시간에 따른 T² 변화율로 대체
#     t2_diff = np.diff(t2_scores)
#     ax2.plot(time_idx[1:], t2_diff, 'k-', alpha=0.7)
#     ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
#     ax2.set_ylabel('T² Change')
#     ax2.set_xlabel('Time')
#     ax2.grid(True, alpha=0.3)
    
#     # x축 포맷
#     for ax in [ax1, ax2]:
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
#         plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
#     plt.tight_layout()
#     plt.show()

# # 선택된 태그에 대해 상세 플롯
# if results_improved:
#     selected_tag = list(results_improved.keys())[0]
#     plot_t2_with_details(selected_tag, results_improved[selected_tag])
