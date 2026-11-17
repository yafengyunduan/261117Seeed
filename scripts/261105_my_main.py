"""独立模式入口；电梯识别共用elevator.panel_261112，抓取/动作流程沿用app。"""
from pathlib import Path
import multiprocessing as mp
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elevator.app import main

if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
