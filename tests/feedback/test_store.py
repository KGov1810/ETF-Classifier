import multiprocessing as mp

import pytest
from filelock import FileLock

from etf_classifier.feedback.store import JOURNAL_COLUMNS, FeedbackError, FeedbackStore


def test_record_is_read_back_identically(store, new_etf):
    record = store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    journal = store.read_journal()
    assert list(journal.columns) == JOURNAL_COLUMNS
    row = journal.iloc[0]
    assert row["ValidationId"] == record["ValidationId"]
    assert row["Name"] == "New Clean Energy"
    assert row["Composition"] == "[{'ENPH': 6.0}, {'FSLR': 5.0}, {'VWS': 4.0}]"
    assert row["Category"] == "['Clean Energy']"
    assert row["ValidatedBy"] == "Kevin"


def test_header_is_written_only_once(store, new_etf):
    for i in range(3):
        store.record_validation({**new_etf, "name": f"ETF {i}"}, ["Clean Energy"], validated_by="Kevin")
    text = store.journal_path.read_text(encoding="utf-8")
    assert text.count("ValidationId") == 1
    assert len(store.read_journal()) == 3


def test_validation_ids_are_unique(store, new_etf):
    ids = {store.record_validation({**new_etf, "name": f"ETF {i}"}, ["Gold"], "Kevin")["ValidationId"]
           for i in range(20)}
    assert len(ids) == 20


def _worker(args):
    settings, worker, n = args
    store = FeedbackStore(settings)
    for i in range(n):
        store.record_validation({"name": f"W{worker}-{i}", "description": "concurrent " * 20,
                                 "composition": {"AAPL": 1.0 + i}}, ["US Equity"], validated_by=f"user{worker}")


def test_concurrent_writers_lose_and_mix_nothing(fb_settings):
    with mp.get_context("spawn").Pool(4) as pool:
        # timeout: a worker that cannot start (e.g. import error) must fail the test, not hang it
        pool.map_async(_worker, [(fb_settings, w, 15) for w in range(4)]).get(timeout=120)
    journal = FeedbackStore(fb_settings).read_journal()
    assert len(journal) == 60
    assert journal["ValidationId"].is_unique
    assert set(journal["Name"]) == {f"W{w}-{i}" for w in range(4) for i in range(15)}
    assert all(journal["Category"] == "['US Equity']")


def test_unavailable_lock_raises_a_clear_error_without_writing(store, new_etf, fb_settings):
    store.lock_timeout = 0.2
    with FileLock(str(store.journal_path) + ".lock"):
        with pytest.raises(FeedbackError, match="try again"):
            store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    assert not store.journal_path.exists() or store.read_journal().empty
