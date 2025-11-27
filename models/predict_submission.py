# models/predict_submission.py

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from ids.io_utils import load_csv_with_meta
from ids.windowing import make_time_windows
from ids.features import window_to_feature_vector


def majority_vote(labels: List[str]) -> str:
    """라벨 리스트에서 다수결 라벨 반환. 비어 있으면 Normal."""
    if not labels:
        return "Normal"
    return Counter(labels).most_common(1)[0][0]


def smooth_labels_global(seq: List[str], window: int = 3) -> List[str]:
    """
    전체 시퀀스에 대해 단순 global smoothing.
    각 index i에 대해 [i-window+1, i] 구간 majority label 적용.
    """
    out: List[str] = []
    for i in range(len(seq)):
        start = max(0, i - window + 1)
        out.append(majority_vote(seq[start : i + 1]))
    return out


def smooth_labels_by_id(labels: List[str], ids: List[str], window: int = 3) -> List[str]:
    """
    같은 CAN ID끼리만 따로 묶어서 smoothing 하는 ID-aware smoothing.
    """
    labels = list(labels)
    df_idx = pd.DataFrame({"id": ids, "label": labels})

    for _, group in df_idx.groupby("id").groups.items():
        idxs = sorted(list(group))
        local = [labels[i] for i in idxs]

        smoothed = []
        for j in range(len(local)):
            start = max(0, j - window + 1)
            smoothed.append(majority_vote(local[start : j + 1]))

        for j, idx in enumerate(idxs):
            labels[idx] = smoothed[j]

    return labels


def replay_spoofing_heuristic(labels: List[str], window: int = 3) -> List[str]:
    """
    Replay / Spoofing 구간을 가볍게 후처리하는 heuristic.
    - 주변 window 내 Replay가 자기 하나뿐이고 Spoofing/Normal이 더 많으면 조정.
    """
    out = labels[:]
    n = len(out)
    for i in range(n):
        if out[i] != "Replay":
            continue

        start = max(0, i - window)
        end = min(n, i + window + 1)
        ctx = out[start:end]
        cnt = Counter(ctx)

        if cnt["Replay"] == 1:
            if cnt["Spoofing"] > 0:
                out[i] = "Spoofing"
            elif cnt["Normal"] > 0:
                out[i] = "Normal"
    return out


def main():
    parser = argparse.ArgumentParser(description="2-Stage IDS Submission Generator")
    parser.add_argument("--test-csv", required=True, help="라벨 없는 test CSV 경로")
    parser.add_argument("--template-csv", required=True, help="submission_template.csv 경로")
    parser.add_argument("--window-sec", type=float, default=0.2, help="슬라이딩 윈도우 크기 (초)")
    parser.add_argument("--binary-model", default="models_artifacts/binary_rf.pkl", help="Binary 모델 pkl 경로")
    parser.add_argument("--attack-model", default="models_artifacts/attack_rf.pkl", help="Attack 4-class 모델 pkl 경로")
    parser.add_argument("--out", default="submission.csv", help="저장할 submission CSV 경로")
    parser.add_argument(
        "--bin-threshold",
        type=float,
        default=0.85,
        help="Binary Attack 판정 threshold (p>=threshold → Attack)",
    )
    args = parser.parse_args()

    print(f"[1] Loading test CSV: {args.test_csv}")
    df, col_info = load_csv_with_meta(args.test_csv)
    n_rows = len(df)
    print(f"[1] Test rows: {n_rows}")

    id_col = col_info["id"]
    if id_col is None:
        raise ValueError("CAN ID 컬럼을 찾을 수 없습니다.")
    ids_seq = df[id_col].astype(str).tolist()

    print("[2] Loading models...")
    bin_obj = joblib.load(args.binary_model)
    atk_obj = joblib.load(args.attack_model)

    bin_model = bin_obj["model"]
    atk_model = atk_obj["model"]
    bin_feat_cols = bin_obj["feature_columns"]
    atk_feat_cols = atk_obj["feature_columns"]
    atk_class_map = atk_obj["class_map"]  # int → str

    print("[3] Building time windows...")
    windows = make_time_windows(df, col_info, window_sec=args.window_sec)
    print(f"[3] Windows: {len(windows)}")

    # 윈도우 feature matrix 생성
    feat_list = [window_to_feature_vector(w.df, col_info) for w in windows]
    X_win_full = pd.DataFrame(feat_list).fillna(0.0)

    # Binary / Attack 각각 자기 feature_columns 순서에 맞게 재정렬
    X_bin = X_win_full.reindex(columns=bin_feat_cols, fill_value=0.0)
    X_atk = X_win_full.reindex(columns=atk_feat_cols, fill_value=0.0)

    # ------------------------
    # Stage 1: Binary Normal/Attack (확률 + 높은 threshold)
    # ------------------------
    print("[4] Stage 1 Binary prediction (Normal/Attack)...")
    bin_proba = bin_model.predict_proba(X_bin)[:, 1]  # Attack 확률
    # Attack 비율이 매우 낮으므로 threshold를 0.85로 높게 설정
    bin_pred = np.where(bin_proba >= args.bin_threshold, "Attack", "Normal")

    # ------------------------
    # Stage 2: Attack classification
    # ------------------------
    print("[5] Stage 2 Attack prediction (DoS/Fuzzing/Replay/Spoofing)...")
    final_window_labels: List[str] = []
    for i in range(len(windows)):
        if bin_pred[i] == "Normal":
            final_window_labels.append("Normal")
        else:
            x_row = X_atk.iloc[[i]]
            atk_label_int = atk_model.predict(x_row)[0]
            atk_label = atk_class_map[int(atk_label_int)]
            final_window_labels.append(atk_label)

    # ------------------------
    # window-level → row-level 매핑
    # ------------------------
    print("[6] Map window predictions to each row...")
    per_row_labels: List[List[str]] = [[] for _ in range(n_rows)]
    for w, w_label in zip(windows, final_window_labels):
        for rid in w.row_indices:
            if 0 <= rid < n_rows:
                per_row_labels[rid].append(w_label)

    # Row별 majority vote
    row_final_labels = [majority_vote(lst) for lst in per_row_labels]

    # ------------------------
    # Post-processing (smoothing & heuristic)
    # ------------------------
    print("[7] Global smoothing (window=3)...")
    row_final_labels = smooth_labels_global(row_final_labels, window=3)

    print("[8] ID-aware smoothing (window=3)...")
    row_final_labels = smooth_labels_by_id(row_final_labels, ids_seq, window=3)

    print("[9] Replay/Spoofing heuristic (window=3)...")
    row_final_labels = replay_spoofing_heuristic(row_final_labels, window=3)

    # ------------------------
    # submission_template과 merge
    # ------------------------
    print(f"[10] Loading submission template: {args.template_csv}")
    sub_df = pd.read_csv(args.template_csv)

    if "Label" not in sub_df.columns:
        raise ValueError("submission_template.csv 에 'Label' 컬럼이 필요합니다.")

    if len(sub_df) != n_rows:
        print(
            f"[!] Warning: template rows({len(sub_df)}) != test rows({n_rows}). "
            f"현재 구현은 행 순서 기준으로 Label을 채웁니다."
        )

    # 길이 맞추기
    if len(row_final_labels) < len(sub_df):
        row_final_labels = row_final_labels + ["Normal"] * (len(sub_df) - len(row_final_labels))
    elif len(row_final_labels) > len(sub_df):
        row_final_labels = row_final_labels[: len(sub_df)]

    sub_df["Label"] = row_final_labels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"[✓] Saved submission → {out_path}")


if __name__ == "__main__":
    main()
