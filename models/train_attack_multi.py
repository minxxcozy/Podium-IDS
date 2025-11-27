# models/train_attack_multi.py

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
from ids.features import window_to_feature_vector, label_for_window_attack


def build_attack_dataset(csv_path: str, window_sec: float) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Train CSV → 슬라이딩 윈도우 → Attack-only dataset
    Label: {DoS, Fuzzing, Replay, Spoofing}
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
        y_attack = label_for_window_attack(wdf, label_col)
        if y_attack is None:
            # Normal-only 윈도우 → Stage2 학습에서는 제외
            continue
        feat = window_to_feature_vector(wdf, col_info)
        feat_list.append(feat)
        labels.append(y_attack)

    X = pd.DataFrame(feat_list)
    y_arr = np.array(labels, dtype=str)
    return X, y_arr


def main():
    parser = argparse.ArgumentParser(description="Train 4-class Attack XGBoost model.")
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
        default="models_artifacts/attack_rf.pkl",  # 경로/이름 그대로 유지
        help="저장할 모델 경로",
    )

    args = parser.parse_args()

    csv_path = args.csv
    window_sec = args.window_sec
    out_path = Path(args.out)

    print(f"[1] Loading train CSV: {csv_path}")
    X, y = build_attack_dataset(csv_path, window_sec)
    print(f"[1] Attack-only Windows: {len(X)}, Features: {X.shape[1]}")
    print(f"[1] Label dist:")
    print(pd.Series(y).value_counts())

    if len(X) == 0:
        raise RuntimeError("Attack 윈도우가 0개입니다. Label 구성 / 윈도우 크기를 확인하세요.")

    X = X.fillna(0.0)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classes, counts = np.unique(y_train, return_counts=True)
    n_classes = len(classes)
    print(f"[1] Classes: {classes}, counts: {counts}")

    print("[2] Training XGBoost (4-class Attack)...")
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("[3] Validation report:")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "window_sec": window_sec,
            "feature_columns": X.columns.tolist(),
        },
        out_path,
    )
    print(f"[✓] Saved attack model → {out_path}")


if __name__ == "__main__":
    main()
