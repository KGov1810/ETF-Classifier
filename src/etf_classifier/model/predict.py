"""Propositions de catégories pour de nouveaux ETF (CSV sans colonne Category).

Usage :
    python -m etf_classifier.model.predict --csv nouveaux_etfs.csv [--model current] [--output …]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import load_dataset
from etf_classifier.features.build import announce_embedding_plan, confirm_spending
from etf_classifier.features.embeddings import create_embedder
from etf_classifier.model.registry import NoModelError, load_bundle

EXIT_OK, EXIT_CANCELLED, EXIT_NO_MODEL, EXIT_NO_AZURE = 0, 1, 2, 3


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Propositions de catégories pour de nouveaux ETF")
    parser.add_argument("--csv", required=True, help="CSV des nouveaux ETF (Name;Description;Composition)")
    parser.add_argument("--model", default="current", help="Version du modèle (défaut : modèle courant)")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    parser.add_argument("--output", help="CSV de sortie (défaut : reports/predictions_<version>_<date>.csv)")
    parser.add_argument("--yes", action="store_true", help="Ne pas demander de confirmation avant la dépense Azure")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")

    settings = load_settings(args.config)
    try:
        bundle = load_bundle(args.model, settings)
    except NoModelError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return EXIT_NO_MODEL

    loaded = load_dataset(args.csv, settings, require_categories=False)
    data = loaded.data.reset_index(drop=True)
    if bundle.pipeline.uses_embeddings:
        embedder = create_embedder(settings, client=client)
        if embedder is None:
            print("Erreur : ce modèle utilise les embeddings Azure, mais Azure OpenAI n'est pas configuré "
                  "(voir .env.example).", file=sys.stderr)
            return EXIT_NO_AZURE
        bundle.pipeline.attach_embedder(embedder)
        plan, cost = announce_embedding_plan(settings, data, embedder)
        if not confirm_spending(plan, cost, args.yes, input_fn):
            return EXIT_CANCELLED

    cfg = settings["model"]
    predictions = bundle.model.predict(bundle.pipeline.transform(data).matrix,
                                       cfg["max_categories"], cfg["uncertain_threshold"])
    # Fichier autonome : l'équipe y ajoute une colonne Category, puis l'importe (feedback.import)
    output = pd.DataFrame({
        "Name": data["name"],
        "Description": data["description"],
        "Composition": [str([{t: w} for t, w in comp.items()]) for comp in data["composition_raw"]],
        "PredictedCategories": [str(p.categories) for p in predictions],
        "Scores": [str([{c: round(s, 4)} for c, s in p.scores[: cfg["max_categories"]]]) for p in predictions],
        "Uncertain": [p.uncertain for p in predictions],
        "ModelVersion": bundle.version,
    })
    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = resolve_path(cfg["reports_dir"]) / f"predictions_{bundle.version}_{stamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep=";", index=False, encoding="utf-8")

    n_uncertain = int(output["Uncertain"].sum())
    print(f"{len(output)} ETF traités avec le modèle {bundle.version} · {n_uncertain} incertain(s) · "
          f"{len(loaded.log.exclusions())} ligne(s) exclue(s)")
    print(f"Propositions : {output_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
