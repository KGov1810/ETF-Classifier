from datetime import datetime, timedelta, timezone

import pytest

from etf_classifier.feedback.checks import ValidationError


def test_cancelled_validation_is_no_longer_active(store, new_etf):
    record = store.record_validation(new_etf, ["Énergies propres"], validated_by="Kevin")
    store.cancel_validation(record["ValidationId"], cancelled_by="Marie", comment="erreur de saisie")
    assert store.active_validations().empty
    journal = store.read_journal()
    assert list(journal["Decision"]) == ["sans_proposition", "annulation"]
    assert journal.iloc[1]["CancelsId"] == record["ValidationId"]


def test_cancelling_an_unknown_id_fails(store):
    with pytest.raises(ValidationError, match="inconnue"):
        store.cancel_validation("20990101-000000-zzzz", cancelled_by="Marie")


def test_cancelling_twice_fails(store, new_etf):
    record = store.record_validation(new_etf, ["Énergies propres"], validated_by="Kevin")
    store.cancel_validation(record["ValidationId"], cancelled_by="Marie")
    with pytest.raises(ValidationError, match="déjà annulée"):
        store.cancel_validation(record["ValidationId"], cancelled_by="Marie")


def test_a_cancelled_etf_can_be_validated_again(store, new_etf):
    record = store.record_validation(new_etf, ["Énergies propres"], validated_by="Kevin")
    store.cancel_validation(record["ValidationId"], cancelled_by="Marie")
    store.record_validation(new_etf, ["Énergies propres", "Actions Monde"], validated_by="Marie")
    assert len(store.active_validations()) == 1


def test_one_backup_per_day(store, new_etf):
    for i in range(3):
        store.record_validation({**new_etf, "name": f"ETF {i}"}, ["Or"], validated_by="Kevin")
    backups = list(store.backups_dir.glob("validations_*.csv"))
    assert len(backups) == 1  # le journal existait dès la 2e validation, une seule sauvegarde aujourd'hui


def test_old_backups_are_pruned(store, new_etf, fb_settings):
    store.backups_kept = 3
    store.backups_dir.mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(5):
        (store.backups_dir / f"validations_{(start + timedelta(days=day)):%Y%m%d}.csv").write_text("x")
    store.record_validation(new_etf, ["Or"], validated_by="Kevin")
    store.record_validation({**new_etf, "name": "ETF 2"}, ["Or"], validated_by="Kevin")
    assert len(list(store.backups_dir.glob("validations_*.csv"))) == 3
