"""Import d'un lot de validations faites hors interface.

Format d'entrée : la sortie de `model.predict`, complétée par l'équipe d'une colonne Category
(catégories validées). Chaque ligne passe par les mêmes contrôles qu'une validation unitaire.

Usage :
    python -m etf_classifier.feedback.importer --csv valides.csv --by "Prénom Nom" [--create-categories]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.parsing import ParsingError, parse_categories
from etf_classifier.feedback.checks import ValidationError
from etf_classifier.feedback.store import FeedbackError, FeedbackStore

REQUIRED_COLUMNS = ["Name", "Description", "Composition", "Category"]
EXIT_OK, EXIT_BAD_FILE, EXIT_STORE = 0, 2, 3


def _created_categories(store: FeedbackStore) -> set[str]:
    """Catégories créées par les validateurs (hors catégories historiques)."""
    registry = store.registry.read()
    return set(registry.loc[registry["source"] == "créée", "category"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import d'un lot de validations")
    parser.add_argument("--csv", required=True, help="CSV validé (sortie de predict + colonne Category)")
    parser.add_argument("--by", required=True, help="Nom ou e-mail du validateur")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    parser.add_argument("--create-categories", action="store_true", help="Autoriser la création de catégories")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    path = Path(args.csv)
    if not path.is_file():
        print(f"Erreur : fichier introuvable : {path}", file=sys.stderr)
        return EXIT_BAD_FILE
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"Erreur : colonnes absentes : {', '.join(missing)}", file=sys.stderr)
            return EXIT_BAD_FILE
        rows = [(reader.line_num, row) for row in reader]

    store = FeedbackStore(settings)
    categories_before = _created_categories(store)
    accepted, refused = [], []
    for line, row in rows:
        name = (row.get("Name") or "").strip()
        try:
            categories = parse_categories(row.get("Category") or "")
            predicted_raw = (row.get("PredictedCategories") or "").strip()
            predicted = parse_categories(predicted_raw) if predicted_raw else None
        except ParsingError as exc:
            refused.append((line, name, [str(exc)]))
            continue
        if not categories:
            refused.append((line, name, ["catégorie validée manquante (colonne Category vide)"]))
            continue
        try:
            record = store.record_validation(
                {"name": name, "description": row.get("Description", ""), "composition": row.get("Composition", "")},
                categories, validated_by=args.by, predicted_categories=predicted,
                model_version=(row.get("ModelVersion") or "").strip() or None,
                create_categories=args.create_categories,
            )
            accepted.append((line, name, record["Decision"]))
        except ValidationError as exc:
            refused.append((line, name, exc.errors))
        except FeedbackError as exc:
            print(f"Erreur : {exc} ({len(accepted)} ligne(s) déjà enregistrée(s))", file=sys.stderr)
            return EXIT_STORE
    created = sorted(_created_categories(store) - categories_before)

    lines = [f"# Import de validations : {path.name}", "",
             f"{len(accepted)} validation(s) enregistrée(s) · {len(refused)} ligne(s) refusée(s) · "
             f"{len(created)} catégorie(s) créée(s)", ""]
    if created:
        lines += [f"Catégories créées : {', '.join(created)}", ""]
    if refused:
        lines += ["## Lignes refusées", "", "| Ligne | ETF | Raison |", "|---|---|---|"]
        lines += [f"| {line} | {name} | {' ; '.join(errors)} |" for line, name, errors in refused]
        lines.append("")
    report = "\n".join(lines)
    target = resolve_path(settings["model"]["reports_dir"]) / f"feedback_import_{datetime.now():%Y%m%d-%H%M%S}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"Compte rendu : {target}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
