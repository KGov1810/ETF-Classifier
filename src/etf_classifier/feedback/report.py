"""Rapport de suivi des validations : volume, taux d'acceptation, alerte de biais, faiblesses du modèle.

Un taux d'acceptation sans modification proche de 100 % sur beaucoup de validations peut signifier que
les propositions sont validées sans être relues : le modèle renforcerait alors ses propres erreurs.

Usage :
    python -m etf_classifier.feedback.report [--since 2026-10-01]
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date

import pandas as pd

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.parsing import parse_categories
from etf_classifier.feedback.store import ACCEPTED, CANCELLATION, CORRECTED, FeedbackStore


@dataclass
class FeedbackReport:
    markdown: str
    stats: dict


def _rate(group: pd.DataFrame) -> tuple[int, float | None]:
    proposed = group[group["Decision"].isin([ACCEPTED, CORRECTED])]
    if proposed.empty:
        return 0, None
    return len(proposed), float((proposed["Decision"] == ACCEPTED).mean())


def build_feedback_report(settings: dict, since: date | None = None) -> FeedbackReport:
    cfg = settings["feedback"]
    store = FeedbackStore(settings)
    journal = store.read_journal() if store.journal_path.exists() else pd.DataFrame(columns=["Decision"])
    if since is not None and not journal.empty:
        journal = journal[journal["ValidatedAt"].str[:10] >= since.isoformat()]
    lines = ["# Suivi des validations", ""]
    empty_stats = {"n_validations": 0, "n_cancelled": 0, "acceptance_rate": None, "by_validator": {},
                   "added": {}, "removed": {}, "bias_alerts": [], "created_categories": []}
    if journal.empty:
        return FeedbackReport("\n".join(lines + ["Aucune validation enregistrée sur cette période.", ""]), empty_stats)

    active = FeedbackStore._active(journal)
    n_cancelled = int((journal["Decision"] == CANCELLATION).sum())
    n_proposed, rate = _rate(active)
    by_validator = {}
    for validator, group in active.groupby("ValidatedBy"):
        n, r = _rate(group)
        by_validator[validator] = {"n_validations": len(group), "n_with_proposal": n, "acceptance_rate": r}

    added, removed = Counter(), Counter()
    for row in active[active["Decision"] == CORRECTED].itertuples():
        validated, predicted = set(parse_categories(row.Category)), set(parse_categories(row.PredictedCategories))
        added.update(validated - predicted)
        removed.update(predicted - validated)

    threshold, minimum = float(cfg["bias_acceptance_threshold"]), int(cfg["bias_min_validations"])
    alerts = [v for v, s in sorted(by_validator.items())
              if s["acceptance_rate"] is not None and s["n_with_proposal"] >= minimum
              and s["acceptance_rate"] >= threshold]
    if rate is not None and n_proposed >= minimum and rate >= threshold:
        alerts.append("global")

    registry = store.registry.read()
    created = registry[registry["source"] == "créée"]
    if since is not None:
        created = created[created["created_at"].str[:10] >= since.isoformat()]
    stats = {"n_validations": len(active), "n_cancelled": n_cancelled, "acceptance_rate": rate,
             "by_validator": by_validator, "added": dict(added), "removed": dict(removed),
             "bias_alerts": alerts, "created_categories": sorted(created["category"])}

    def pct(value):
        return f"{value:.0%}" if value is not None else "–"

    lines += [f"Période : depuis {since or 'le début'}", "",
              "| Indicateur | Valeur |", "|---|---|",
              f"| Validations actives | {len(active)} |",
              f"| Annulations | {n_cancelled} |",
              f"| Taux d'acceptation sans modification | {pct(rate)} (sur {n_proposed} proposition(s)) |", ""]
    for who in alerts:
        label = "l'ensemble des validateurs" if who == "global" else who
        lines.append(f"> ⚠️ Risque de biais pour {label} : au moins {threshold:.0%} des propositions acceptées "
                     f"sans modification (≥ {minimum} validations). Vérifier que les propositions sont bien relues.")
    if alerts:
        lines.append("")
    lines += ["## Par validateur", "", "| Validateur | Validations | Avec proposition | Taux d'acceptation |",
              "|---|---|---|---|"]
    lines += [f"| {v} | {s['n_validations']} | {s['n_with_proposal']} | {pct(s['acceptance_rate'])} |"
              for v, s in sorted(by_validator.items())]
    lines += ["", "## Par jour", "", "| Jour | Validations |", "|---|---|"]
    lines += [f"| {day} | {n} |" for day, n in active["ValidatedAt"].str[:10].value_counts().sort_index().items()]
    lines += ["", "## Corrections des validateurs (faiblesses du modèle)", "",
              "| Catégorie | Ajoutée par les validateurs | Retirée par les validateurs |", "|---|---|---|"]
    for category in sorted(set(added) | set(removed)):
        lines.append(f"| {category} | {added.get(category, 0)} | {removed.get(category, 0)} |")
    lines += ["", "## Catégories créées", "", ", ".join(stats["created_categories"]) or "Aucune.", ""]
    return FeedbackReport("\n".join(lines), stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rapport de suivi des validations")
    parser.add_argument("--since", type=date.fromisoformat, help="Date de début (AAAA-MM-JJ)")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    report = build_feedback_report(settings, args.since)
    path = resolve_path(settings["model"]["reports_dir"]) / "feedback.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"Rapport : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
