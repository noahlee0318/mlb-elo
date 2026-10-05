"""Bundle existing public MLB data for a fresh Streamlit Cloud checkout.

Run: python -m scripts.bundle_deployment_data --source-data data
Rebuild the source CSVs with the documented pipeline before refreshing bundles.
"""

import argparse
import gzip
from pathlib import Path
import shutil

from src.deployment_data import BOOTSTRAP_FILES, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    for name in BOOTSTRAP_FILES:
        if not (args.source_data / name).is_file():
            raise FileNotFoundError(args.source_data / name)
    destination = DATA_DIR / "bootstrap"
    destination.mkdir(exist_ok=True)
    for name in BOOTSTRAP_FILES:
        with (args.source_data / name).open("rb") as source:
            with (destination / f"{name}.gz").open("wb") as target:
                with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as packed:
                    shutil.copyfileobj(source, packed)
        print(f"Bundled {name}: {(destination / f'{name}.gz').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
