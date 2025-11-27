# ids/config.py

"""
CSV 컬럼 이름 후보 설정

실제 autohack2025_train.csv / test_data.csv 컬럼 이름에 맞게
필요하면 여기 후보 리스트를 수정하면 됨.
"""

TIMESTAMP_CANDIDATES = ["Timestamp", "Time", "time", "timestamp"]
ID_CANDIDATES = ["Arbitration_ID", "CAN_ID", "ID", "can_id"]
DLC_CANDIDATES = ["DLC", "dlc", "Length"]
DATA_CANDIDATES = ["Data", "Payload", "data", "payload"]
LABEL_CANDIDATES = ["Label", "label", "Class", "Attack", "attack", "category"]