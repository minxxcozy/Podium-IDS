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
    """
    라벨 리스트에서 다수결 라벨 반환.
    비어 있으면 Normal.
    """
    if not labels:
        return "Normal"
    c = Counter(labels)
    return c.most_common(1)[0][0]


def smooth_labels_global(seq: List[str], window: int = 5) -> List[str]:
    """
    전체 시퀀스에 대해 단순 global smoothing.
    각 index i에 대해 [i-window+1, i] 구간 majority label 적용.
    """
    n = len(seq)
    out = []
    for i in range(n):
        start = max(0, i - window + 1)
        sub = seq[start : i + 1]
        out.append(majority_vote(sub))
    return out


def smooth_labels_by_id(
    labels: List[str],
    ids: List[str],
    window: int = 5,
) -> List[str]:
    """
    같은 CAN ID끼리만 따로 묶어서 smoothing 하는 ID-aware smoothing.
    - ID 별로 시계열 순서를 유지한 상태에서 majority smoothing.
    """
    labels = list(labels)
    n = len(labels)
    if n == 0:
        return labels

    df_idx = pd.DataFrame({"id": ids, "label": labels})
    # ID별로 index 리스트를 모아서 smoothing
    for can_id, group in df_idx.groupby("id").groups.items():
        idx_list = sorted(list(group))
        local_labels = [labels[i] for i in idx_list]

        # local 시퀀스 smoothing
        smoothed_local = []
        for j in range(len(local_labels)):
            start = max(0, j - window + 1)
            sub = local_labels[start : j + 1]
            smoothed_local.append(majority_vote(sub))

        # 다시 원래 index에 반영
        for j, idx in enumerate(idx_list):
            labels[idx] = smoothed_local[j]

    return labels


def replay_spoofing_heuristic(labels: List[str], window: int = 5) -> List[str]:
    """
    Replay / Spoofing 구간을 가볍게 후처리하는 heuristic.

    아이디어:
    - Replay는 보통 일정 구간 연속으로 나타나는 경향이 있음.
      → 주변 window 범위에서 자신만 Replay이면 과한 False Positive일 가능성이 높음.
      → 이 경우 Spoofing 또는 Normal로 조정.
    - 두 클래스는 Weighted F1 비중이 높으므로 (0.3 + 0.3),
      잘못된 단일 Replay peak를 줄여 점수 안정성 확보.
    """
    out = list(labels)
    n = len(out)
    for i in range(n):
        if out[i] != "Replay":
            continue
        # 주변 window 내 label 통계
        start = max(0, i - window)
        end = min(n, i + window + 1)
        ctx = out[start:end]
        cnt = Counter(ctx)
        # 이 구간 내 Replay가 자기 하나 뿐이고, Spoofing이 더 많다면 Spoofing으로 조정
        if cnt["Replay"] == 1:
            if cnt["Spoofing"] > 0:
                out[i] = "Spoofing"
            elif cnt["Normal"] > 0 and cnt["Spoofing"] == 0:
                # 주변이 전부 Normal이면, 공격으로 볼 근거가 약하다고 보고 Normal로
                out[i] = "Normal"
    return out


def main():
    parser = argparse.ArgumentParser(description="2-Stage IDS로 test CSV 예측 후 submission.csv 생성")

    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="라벨이 없는 test CSV (예: data/autohack2025_test_data.csv)",
    )
    parser.add_argument(
        "--template-csv",
        type=str,
        required=True,
        help="submission_template.csv 경로",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.2,
        help="슬라이딩 윈도우 크기 (초) - 학습 시 사용한 값과 동일해야 함",
    )
    parser.add_argument(
        "--binary-model",
        type=str,
        default="models_artifacts/binary_rf.pkl",
        help="Binary 모델 pkl 경로",
    )
    parser.add_argument(
        "--attack-model",
        type=str,
        default="models_artifacts/attack_rf.pkl",
        help="Attack 4-class 모델 pkl 경로",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="저장할 submission CSV 경로",
    )

    args = parser.parse_args()

    test_csv = args.test_csv
    template_csv = args.template_csv
    window_sec = args.window_sec
    out_path = Path(args.out)

    print(f"[1] Loading test CSV: {test_csv}")
    df, col_info = load_csv_with_meta(test_csv)

    n_rows = len(df)
    print(f"[1] Test rows: {n_rows}")

    id_col = col_info["id"]
    if id_col is None:
        raise ValueError("CAN ID 컬럼을 찾을 수 없습니다. config.ID_CANDIDATES를 확인하세요.")
    ids_seq = df[id_col].astype(str).tolist()

    print("[2] Loading models...")
    bin_obj = joblib.load(args.binary_model)
    atk_obj = joblib.load(args.attack_model)
    bin_model = bin_obj["model"]
    atk_model = atk_obj["model"]

    print("[3] Building time windows...")
    windows = make_time_windows(df, col_info, window_sec=window_sec)
    print(f"[3] Windows: {len(windows)}")

    # 윈도우 feature matrix 생성
    feat_list = []
    for w in windows:
        feat = window_to_feature_vector(w.df, col_info)
        feat_list.append(feat)

    X_win = pd.DataFrame(feat_list)
    X_win = X_win.fillna(0.0)

    print("[4] Stage 1: Binary Normal/Attack 예측...")
    bin_pred = bin_model.predict(X_win)  # shape: (n_windows,)

    print("[5] Stage 2: Attack-class 예측...")
    final_window_labels: List[str] = []
    for i in range(len(windows)):
        if bin_pred[i] == "Normal":
            final_window_labels.append("Normal")
        else:
            x_row = X_win.iloc[[i]]
            atk_label = atk_model.predict(x_row)[0]
            final_window_labels.append(str(atk_label))

    # window-level prediction → row-level prediction (row index 기준)
    print("[6] Map window predictions to each row...")
    per_row_labels: List[List[str]] = [[] for _ in range(n_rows)]

    for w, w_label in zip(windows, final_window_labels):
        for rid in w.row_indices:
            if 0 <= rid < n_rows:
                per_row_labels[rid].append(w_label)

    # 각 row별 majority vote
    row_final_labels = [majority_vote(lst) for lst in per_row_labels]

    # 1차 global smoothing
    print("[7] Global smoothing...")
    row_final_labels = smooth_labels_global(row_final_labels, window=5)

    # 2차 ID-aware smoothing
    print("[8] ID-aware smoothing...")
    row_final_labels = smooth_labels_by_id(row_final_labels, ids_seq, window=5)

    # 3차 Replay/Spoofing heuristic
    print("[9] Replay/Spoofing heuristic post-processing...")
    row_final_labels = replay_spoofing_heuristic(row_final_labels, window=5)

    # submission_template과 merge
    print(f"[10] Loading submission template: {template_csv}")
    sub_df = pd.read_csv(template_csv)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"[✓] Saved submission → {out_path}")


if __name__ == "__main__":
    main()