# ids/windowing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class TimeWindow:
    df: pd.DataFrame           # 이 윈도우에 포함된 행들
    start_time: float
    end_time: float
    row_indices: np.ndarray    # 원본 df에서의 행 인덱스 (0 ~ N-1, 정수)


def normalize_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Timestamp를 0부터 시작하도록 정규화."""
    df = df.copy()
    df[ts_col] = df[ts_col] - df[ts_col].min()
    return df


def make_time_windows(
    df: pd.DataFrame,
    col_info: Dict[str, Optional[str]],
    window_sec: float,
    step_sec: Optional[float] = None,
) -> List[TimeWindow]:
    """
    슬라이딩 윈도우 생성.

    window_sec: 윈도우 길이 (초)
    step_sec: 슬라이딩 스텝 (미지정 시 50% overlap)

    반환: TimeWindow 리스트
    """
    ts_col = col_info["timestamp"]
    if ts_col is None:
        raise ValueError("timestamp 컬럼 정보가 없습니다.")

    if step_sec is None:
        step_sec = window_sec / 2.0

    # 원본 인덱스를 row_id로 별도 보존
    df = df.copy()
    df = df.sort_values(ts_col).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=int)

    df = normalize_timestamp(df, ts_col)

    ts = df[ts_col].values
    max_t = float(ts.max())

    windows: List[TimeWindow] = []
    cur = 0.0

    while cur <= max_t:
        start = cur
        end = cur + window_sec

        mask = (df[ts_col] >= start) & (df[ts_col] < end)
        wdf = df[mask]

        if len(wdf) > 0:
            windows.append(
                TimeWindow(
                    df=wdf.drop(columns=["_row_id"]),
                    start_time=start,
                    end_time=end,
                    row_indices=wdf["_row_id"].to_numpy(),
                )
            )

        cur += step_sec

    return windows
