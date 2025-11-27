# ids/features.py

from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np
import pandas as pd


def hex_payload_to_bytes(hex_str: str) -> List[int]:
    """
    "00 1A FF 20 ..." 형태의 hex string을 byte 리스트로 변환.
    공백/탭/콤마 등은 무시.
    """
    if not isinstance(hex_str, str):
        return []

    cleaned = (
        hex_str.replace("\t", " ")
        .replace(",", " ")
        .replace(";", " ")
        .strip()
    )
    if not cleaned:
        return []

    parts = [p for p in cleaned.split(" ") if p]
    bytes_list: List[int] = []
    for p in parts:
        try:
            bytes_list.append(int(p, 16))
        except ValueError:
            # 이상한 값 들어있으면 무시
            continue
    return bytes_list


def shannon_entropy(values: List[int]) -> float:
    """간단한 Shannon entropy 계산."""
    if not values:
        return 0.0
    arr = np.asarray(values)
    values, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def byte_histogram_features(byte_values: List[int], n_bins: int = 16) -> Dict[str, float]:
    """
    0~255 byte 값을 n_bins 개의 구간으로 나눈 히스토그램 feature.
    예: n_bins=16 → bin당 16개 값.
    """
    feats: Dict[str, float] = {}
    if not byte_values:
        for i in range(n_bins):
            feats[f"byte_bin_{i}_ratio"] = 0.0
        return feats

    arr = np.asarray(byte_values, dtype=float)
    hist, _ = np.histogram(arr, bins=n_bins, range=(0, 256))
    total = hist.sum()
    if total <= 0:
        for i in range(n_bins):
            feats[f"byte_bin_{i}_ratio"] = 0.0
        return feats

    ratios = hist / total
    for i in range(n_bins):
        feats[f"byte_bin_{i}_ratio"] = float(ratios[i])
    return feats


def bit_level_entropy(byte_values: List[int]) -> float:
    """
    byte 리스트를 bit 시퀀스로 펼친 후 Shannon entropy 계산.
    (0/1 분포)
    """
    if not byte_values:
        return 0.0
    bits: List[int] = []
    for b in byte_values:
        v = int(b) & 0xFF
        for shift in range(8):
            bits.append((v >> shift) & 1)
    return shannon_entropy(bits)


def window_to_feature_vector(
    wdf: pd.DataFrame,
    col_info: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """
    한 윈도우(df)에 대해 고급 feature vector 생성.
    (학습/예측 양쪽에서 공통으로 사용)
    """
    ts_col = col_info["timestamp"]
    id_col = col_info["id"]
    dlc_col = col_info["dlc"]
    data_col = col_info["data"]

    feats: Dict[str, float] = {}

    n_msgs = len(wdf)
    feats["n_msgs"] = float(n_msgs)

    if n_msgs == 0:
        base_keys = [
            "n_ids",
            "msgs_per_sec",
            "duration",
            "dt_mean",
            "dt_std",
            "dt_min",
            "dt_max",
            "id_entropy",
            "id_max_freq",
            "id_top1_freq",
            "id_top2_freq",
            "id_top3_freq",
            "dlc_mean",
            "dlc_std",
            "payload_mean",
            "payload_std",
            "payload_entropy",
            "payload_bit_entropy",
        ]
        for k in base_keys:
            feats[k] = 0.0
        feats.update(byte_histogram_features([], n_bins=16))
        return feats

    # 시간 관련
    ts = wdf[ts_col].values.astype(float)
    duration = float(ts.max() - ts.min()) if n_msgs > 1 else 0.0
    feats["duration"] = duration
    feats["msgs_per_sec"] = float(n_msgs) / duration if duration > 0 else float(n_msgs)

    if n_msgs > 1:
        dts = np.diff(ts)
        feats["dt_mean"] = float(dts.mean())
        feats["dt_std"] = float(dts.std())
        feats["dt_min"] = float(dts.min())
        feats["dt_max"] = float(dts.max())
    else:
        feats["dt_mean"] = 0.0
        feats["dt_std"] = 0.0
        feats["dt_min"] = 0.0
        feats["dt_max"] = 0.0

    # ID 관련
    id_series = wdf[id_col]
    id_counts = id_series.value_counts()
    feats["n_ids"] = float(id_counts.shape[0])

    total = float(n_msgs)
    id_freqs = (id_counts / total).values.tolist()
    for i in range(3):
        feats[f"id_top{i+1}_freq"] = float(id_freqs[i]) if i < len(id_freqs) else 0.0

    feats["id_entropy"] = shannon_entropy(id_series.astype("category").cat.codes.tolist())
    feats["id_max_freq"] = float(id_freqs[0]) if id_freqs else 0.0

    # DLC
    if dlc_col is not None and dlc_col in wdf.columns:
        dlc_vals = wdf[dlc_col].values.astype(float)
        feats["dlc_mean"] = float(dlc_vals.mean())
        feats["dlc_std"] = float(dlc_vals.std())
    else:
        feats["dlc_mean"] = 0.0
        feats["dlc_std"] = 0.0

    # Payload
    all_bytes: List[int] = []
    for s in wdf[data_col].values:
        all_bytes.extend(hex_payload_to_bytes(s))

    if all_bytes:
        b_arr = np.array(all_bytes, dtype=float)
        feats["payload_mean"] = float(b_arr.mean())
        feats["payload_std"] = float(b_arr.std())
        feats["payload_entropy"] = shannon_entropy(all_bytes)
        feats["payload_bit_entropy"] = bit_level_entropy(all_bytes)
    else:
        feats["payload_mean"] = 0.0
        feats["payload_std"] = 0.0
        feats["payload_entropy"] = 0.0
        feats["payload_bit_entropy"] = 0.0

    feats.update(byte_histogram_features(all_bytes, n_bins=16))

    return feats


def label_for_window_binary(
    wdf: pd.DataFrame,
    label_col: str,
) -> str:
    """
    Stage 1용 Label: Normal vs Attack.
    윈도우 안에 하나라도 Normal이 아닌 라벨이 있으면 Attack.
    """
    labels = wdf[label_col].astype(str).values
    if len(labels) == 0:
        return "Normal"
    is_attack = labels != "Normal"
    return "Attack" if is_attack.any() else "Normal"


def label_for_window_attack(
    wdf: pd.DataFrame,
    label_col: str,
) -> Optional[str]:
    """
    Stage 2용 Label: {DoS, Fuzzing, Replay, Spoofing} 중 majority.
    Normal 제외, Attack 라벨만 대상으로 majority vote.
    Attack이 없으면 None.
    """
    labels = wdf[label_col].astype(str).values
    attack_labels = labels[labels != "Normal"]
    if len(attack_labels) == 0:
        return None
    vc = pd.Series(attack_labels).value_counts()
    return str(vc.index[0])