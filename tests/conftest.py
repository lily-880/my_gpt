"""让 `pytest` 能 import 到仓库根目录下的 my_gpt 包。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
