# models/train_binary.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # 🔥 고성능 트리 기반 모델

from ids.io_utils import load_csv_with_meta
from ids.windowing import make_time_windows
from ids.features import window_to_feature_vector, label_for_window_binary


def build_binary_dataset(csv_path: str, window_sec: float) -> tuple[pd.DataFrame, np.ndarray]:
    """
    CSV → 슬라이딩 윈도우 → feature vector + Binary Label (Normal/Attack)
    """
    df, col_info = load_csv_with_meta(csv_path)

    label_col = col_info["label"]
    if label_col is None:
        raise ValueError("Train CSV에 Label 컬럼이 없습니다. config.LABEL_CANDIDATES를 확인하세요.")

    windows = make_time_windows(df, col_info, window_sec=window_sec)

    feat_list = []
    labels = []

    for w in windows:
        wdf = w.df
        y = label_for_window_binary(wdf, label_col)
        feat = window_to_feature_vector(wdf, col_info)

        feat_list.append(feat)
        labels.append(y)

    X = pd.DataFrame(feat_list)
    y_arr = np.array(labels, dtype=str)
    return X, y_arr


def main():
    parser = argparse.ArgumentParser(description="Train Binary Normal/Attack XGBoost model.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Train CSV 경로 (예: data/autohack2025_train.csv)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.2,
        help="슬라이딩 윈도우 크기 (초)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models_artifacts/binary_rf.pkl",  # 경로/이름 그대로 유지
        help="저장할 모델 경로",
    )

    args = parser.parse_args()

    csv_path = args.csv
    window_sec = args.window_sec
    out_path = Path(args.out)

    print(f"[1] Loading train CSV: {csv_path}")
    X, y = build_binary_dataset(csv_path, window_sec)
    print(f"[1] Windows: {len(X)}, Features: {X.shape[1]}")

    # NaN 방지
    X = X.fillna(0.0)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 클래스 비율 → Attack 비율에 맞는 scale_pos_weight 설정
    n_normal = (y_train == "Normal").sum()
    n_attack = (y_train == "Attack").sum()
    if n_attack == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = n_normal / max(1, n_attack)

    print("[2] Training XGBoost (Binary Normal vs Attack)...")
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("[3] Validation report:")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    # 모델 저장 (기존 구조 유지: dict에 model/window_sec/feature_columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "window_sec": window_sec,
            "feature_columns": X.columns.tolist(),
        },
        out_path,
    )
    print(f"[✓] Saved binary model → {out_path}")


if __name__ == "__main__":
    main()
