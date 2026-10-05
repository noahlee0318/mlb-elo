"""Restore bundled public baseball data on a fresh hosted checkout."""

import gzip
from pathlib import Path
import shutil
import tempfile
import threading

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BOOTSTRAP_FILES = ("features.csv", "pitcher_games.csv", "batter_games.csv")
_LOCK = threading.Lock()


def ensure_deployment_data(data_dir=DATA_DIR):
    """Seed missing CSVs only; never replace data updated by a live refresh.

    Decompress to a temporary file first so failed or concurrent starts cannot
    leave a partial CSV for the model pipeline to read.
    """
    data_dir = Path(data_dir)
    with _LOCK:
        for name in BOOTSTRAP_FILES:
            target = data_dir / name
            source = data_dir / "bootstrap" / f"{name}.gz"
            if target.exists() or not source.exists():
                continue
            with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                try:
                    with gzip.open(source, "rb") as packed:
                        shutil.copyfileobj(packed, output)
                except Exception:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if not target.exists():
                    temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
