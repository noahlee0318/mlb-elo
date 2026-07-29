"""Root conftest: anchors pytest's rootdir at the project root and puts it
on sys.path, so tests can `from src.<module> import ...` regardless of the
directory pytest is invoked from or its import mode.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
