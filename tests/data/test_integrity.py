import hashlib
import shutil

from etf_classifier.data.report import main


def _fingerprint(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_csv_is_never_modified(sample_csv, tmp_path):
    source = tmp_path / "etfs.csv"
    shutil.copy(sample_csv, source)
    before = _fingerprint(source)
    main(["--csv", str(source), "--output", str(tmp_path / "rapport.md")])
    assert _fingerprint(source) == before
