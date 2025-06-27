# config/path.py
import os
from pathlib import Path

# 获取项目根目录（自动识别）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 定义常用路径
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
STRATEGY_DIR = PROJECT_ROOT / "strategies"
REPORT_DIR = PROJECT_ROOT / "reports"

def ensure_dirs():
    for path in [DATA_DIR, NOTEBOOK_DIR, STRATEGY_DIR, REPORT_DIR]:
        os.makedirs(path, exist_ok=True)
