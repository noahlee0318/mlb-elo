import gzip

import pytest

from src.deployment_data import ensure_deployment_data


def test_restore_preserves_refreshed_data(tmp_path):
    bundle = tmp_path / "bootstrap"
    bundle.mkdir()
    (bundle / "features.csv.gz").write_bytes(gzip.compress(b"feature,value\na,1\n"))
    ensure_deployment_data(tmp_path)
    assert (tmp_path / "features.csv").read_bytes() == b"feature,value\na,1\n"
    (tmp_path / "features.csv").write_bytes(b"updated data")
    ensure_deployment_data(tmp_path)
    assert (tmp_path / "features.csv").read_bytes() == b"updated data"


def test_corrupt_bundle_does_not_publish_partial_csv(tmp_path):
    bundle = tmp_path / "bootstrap"
    bundle.mkdir()
    (bundle / "features.csv.gz").write_bytes(b"broken gzip")
    with pytest.raises(gzip.BadGzipFile):
        ensure_deployment_data(tmp_path)
    assert not (tmp_path / "features.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))
