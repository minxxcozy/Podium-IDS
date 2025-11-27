# ids/io_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from . import config


def _find_column(candidates, columns) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_csv_with_meta(csv_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    CSV를 로드하고, Timestamp / ID / DLC / Data / Label 컬럼을 자동 탐지한다.

    반환:
        df: 원본 DataFrame (읽기 전용으로 다룬다고 생각)
        col_info: {
            "timestamp": <ts_col or None>,
            "id": <id_col or None>,
            "dlc": <dlc_col or None>,
            "data": <data_col or None>,
            "label": <label_col or None>,  # test 데이터는 None일 수 있음
        }
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    ts_col = _find_column(config.TIMESTAMP_CANDIDATES, cols)
    id_col = _find_column(config.ID_CANDIDATES, cols)
    dlc_col = _find_column(config.DLC_CANDIDATES, cols)
    data_col = _find_column(config.DATA_CANDIDATES, cols)
    label_col = _find_column(config.LABEL_CANDIDATES, cols)

    if ts_col is None:
        raise ValueError(f"Timestamp 컬럼을 찾을 수 없음. config.TIMESTAMP_CANDIDATES 를 확인해줘. cols={cols}")
    if id_col is None:
        raise ValueError(f"CAN ID 컬럼을 찾을 수 없음. config.ID_CANDIDATES 를 확인해줘. cols={cols}")
    if data_col is None:
        raise ValueError(f"Data/Payload 컬럼을 찾을 수 없음. config.DATA_CANDIDATES 를 확인해줘. cols={cols}")

    col_info = {
        "timestamp": ts_col,
        "id": id_col,
        "dlc": dlc_col,
        "data": data_col,
        "label": label_col,
    }

    return df, col_info