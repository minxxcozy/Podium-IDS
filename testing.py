import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('submission.csv')

# 모든 가능한 라벨 정의
all_labels = ['DoS', 'Replay', 'Spoofing', 'Fuzzing', 'Normal']

# 각 라벨별 개수 계산 (없는 라벨은 0으로)
label_counts = df['Label'].value_counts().reindex(all_labels, fill_value=0)

# 총합 추가
label_counts['Total'] = label_counts.sum()

# 결과 출력
print(label_counts)