# 🏁 Podium-IDS
> Feature 강화 + XGBoost + 후처리까지 완성된 최고성능 버전

## 🧠 파이프라인
### 1️⃣ Stage 1 — Binary IDS
* Normal / Attack 분류
* XGBoost 기반
* Attack 윈도우만 Stage 2로 전달

### 2️⃣ Stage 2 — Attack 4-Class IDS
* DoS / Fuzzing / Spoofing / Replay
* 중요도 높은 Replay/Spoofing에 특화된 Feature + 후처리

### 3️⃣ 후처리 (Post-processing)
* Global smoothing
* ID-aware smoothing
* Replay/Spoofing heuristic

### 4️⃣ 최종 submission.csv 생성

## 📁 프로젝트 구조
```bash
can-ids/
│
├── data/
│   ├── autohack2025_train.csv
│   ├── autohack2025_test_data.csv
│   └── submission_template.csv
│
├── ids/
│   ├── config.py
│   ├── io_utils.py
│   ├── windowing.py
│   └── features.py
│
├── models/
│   ├── train_binary.py
│   ├── train_attack_multi.py
│   └── predict_submission.py
│
├── models_artifacts/
├── requirements.txt
└── README.md
```

## 🐍 Python 가상환경
### 1️⃣ Python 가상환경 생성
```bash
python3 -m venv .venv
```

### 2️⃣ 가상환경 활성화
```bash
source .venv/bin/activate
```

## ⚙️ requirements 설치
### 1️⃣ pip 최신화
```bash
pip install --upgrade pip
```

### 2️⃣ requirements.txt 설치
```bash
pip install -r requirements.txt
```

## 🧪 파이프라인 실행 명령어
### 1️⃣ Binary 모델 학습
```bash
python3 -m models.train_binary --csv data/autohack2025_train.csv --window-sec 0.02
```

### 2️⃣ Attack 모델 학습
```bash
python3 -m models.train_attack_multi --csv data/autohack2025_train.csv --window-sec 0.02
```

### 3️⃣ Test 데이터 예측 & submission.csv 생성
```bash
python -m models.predict_submission --test-csv data/autohack2025_test_data.csv --template-csv data/submission_template.csv --window-sec 0.02
```
