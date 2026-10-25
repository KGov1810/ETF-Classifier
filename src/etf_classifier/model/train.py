"""Entraînement complet : réglage, évaluation unique sur le test, rapport, sauvegarde, promotion.

Usage :
    python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure        # référence locale
    python -m etf_classifier.model.train --csv data/raw/etfs.csv --yes --promote   # avec embeddings
"""

import argparse
import copy
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import PreparedDataset, prepare_dataset
from etf_classifier.features.build import announce_embedding_plan, confirm_spending
from etf_classifier.features.embeddings import create_embedder
from etf_classifier.features.pipeline import training_fingerprint
from etf_classifier.model.baseline import FrequencyBaseline
from etf_classifier.model.classifier import TuningResult, tune
from etf_classifier.model.evaluate import Metrics, evaluate, render_report
from etf_classifier.model.explain import top_features
from etf_classifier.model.registry import current_version, promote, save_bundle
from etf_classifier.similarity.index import SimilarityIndex

EXIT_OK, EXIT_CANCELLED = 0, 1


@dataclass
class TrainingOutcome:
    result: TuningResult
    metrics: Metrics
    baseline: Metrics
    top: dict
    unknown_categories: dict[str, int]
    seconds: float
    azure_cost_usd: float


def train_and_evaluate(prepared: PreparedDataset, settings: dict, embedder=None) -> TrainingOutcome:
    """Règle le modèle sur l'entraînement seul, puis l'évalue une seule fois sur le test."""
    start = time.perf_counter()
    cost_before = embedder.run_cost_usd if embedder else 0.0
    result = tune(prepared.split.train, settings, embedder)
    cfg = settings["model"]
    test_features = result.pipeline.transform(prepared.split.test)
    y_test = result.pipeline.transform_labels(prepared.split.test)
    unknown = dict(result.pipeline.labels.last_unknown)
    y_pred = result.model.decide(result.model.predict_proba(test_features.matrix), cfg["max_categories"])
    classes = result.model.classes
    baseline = FrequencyBaseline().fit(result.pipeline.transform_labels(prepared.split.train))
    return TrainingOutcome(
        result=result,
        metrics=evaluate(y_test, y_pred, classes),
        baseline=evaluate(y_test, baseline.predict(len(y_test)), classes),
        top=top_features(result.model, test_features.feature_names, k=10),
        unknown_categories=unknown,
        seconds=time.perf_counter() - start,
        azure_cost_usd=(embedder.run_cost_usd if embedder else 0.0) - cost_before,
    )


def _model_card(version: str, outcome: TrainingOutcome, info: dict) -> str:
    m, b = outcome.metrics, outcome.baseline
    lines = [f"# Fiche du modèle {version}", "",
             "Classifieur multi-étiquettes d'ETF : une régression logistique par catégorie, seuils réglés par catégorie.", ""]
    lines += [f"- **{k}** : {v}" for k, v in info.items()]
    lines += ["", "## Performance (jeu de test)", "",
              f"- F1 macro : {m.f1_macro:.3f} (référence triviale : {b.f1_macro:.3f})",
              f"- F1 micro : {m.f1_micro:.3f} (référence triviale : {b.f1_micro:.3f})", "",
              "## Limites", "",
              "- Ne peut pas prédire une catégorie absente de l'entraînement.",
              "- Les catégories rares ont un seuil par défaut quand la validation compte trop peu d'exemples.",
              "- Les performances dépendent de la couverture des compositions (souvent partielles)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Entraînement du classifieur multi-étiquettes")
    parser.add_argument("--csv", required=True, help="Chemin du CSV d'ETF catégorisés")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    parser.add_argument("--no-azure", action="store_true", help="Blocs locaux uniquement (sans embeddings)")
    parser.add_argument("--yes", action="store_true", help="Ne pas demander de confirmation avant la dépense Azure")
    parser.add_argument("--promote", action="store_true", help="Désigner ce modèle comme modèle courant")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")

    settings = load_settings(args.config)
    if args.no_azure:
        settings = copy.deepcopy(settings)
        settings["features"]["blocks"]["embeddings"]["enabled"] = False
    prepared = prepare_dataset(args.csv, settings)
    embedder = None
    if settings["features"]["blocks"]["embeddings"]["enabled"]:
        embedder = create_embedder(settings, client=client)
        if embedder is not None:
            plan, cost = announce_embedding_plan(settings, prepared.data, embedder)
            if not confirm_spending(plan, cost, args.yes, input_fn):
                return EXIT_CANCELLED

    outcome = train_and_evaluate(prepared, settings, embedder)
    result = outcome.result
    first_model = current_version(settings) is None
    info = {
        "ETF d'entraînement / de test": f"{len(prepared.split.train)} / {len(prepared.split.test)}",
        "Empreinte du jeu d'entraînement": training_fingerprint(prepared.split.train),
        "Blocs de variables": ", ".join(result.pipeline.active_blocks),
        "C retenu (validation)": result.best_c,
        "F1 macro de validation par C": ", ".join(f"{c} → {s:.3f}" for c, s in result.validation_scores.items()),
        "Catégories apprises": len(result.model.classes),
        "Durée d'entraînement": f"{outcome.seconds:.1f} s",
        "Coût Azure de l'entraînement": f"{outcome.azure_cost_usd:.4f} $",
    }
    thresholds = dict(zip(result.model.classes, result.model.thresholds.tolist()))
    metrics = {**outcome.metrics.to_dict(), "baseline": outcome.baseline.to_dict(), "best_c": result.best_c,
               "validation_scores": result.validation_scores, "thresholds": thresholds,
               "blocks": result.pipeline.active_blocks, "unknown_categories": outcome.unknown_categories,
               "n_train": len(prepared.split.train), "n_test": len(prepared.split.test),
               "azure_cost_usd": outcome.azure_cost_usd}
    # index de similarité sur toutes les lignes exploitables, rangé avec le modèle (toujours cohérents)
    similarity = SimilarityIndex.build(prepared.data, result.pipeline, settings)
    info["Index de similarité"] = f"{len(prepared.data)} ETF, {similarity.size_mb:.1f} Mo"
    path = save_bundle(result.model, result.pipeline, metrics, settings, "", similarity=similarity)
    version = path.name
    (path / "MODEL_CARD.md").write_text(_model_card(version, outcome, info), encoding="utf-8")
    report = render_report(version, outcome.metrics, outcome.baseline, outcome.top, thresholds,
                           outcome.unknown_categories, info)
    report_path = resolve_path(settings["model"]["reports_dir"]) / f"model_{version}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if args.promote and not first_model:
        promote(version, settings)

    m, b = outcome.metrics, outcome.baseline
    print(f"\nModèle {version} · blocs : {', '.join(result.pipeline.active_blocks)} · C = {result.best_c}")
    print(f"F1 macro : {m.f1_macro:.3f} (référence triviale {b.f1_macro:.3f})")
    print(f"F1 micro : {m.f1_micro:.3f} (référence triviale {b.f1_micro:.3f})")
    if embedder is not None:
        print(f"Coût Azure de l'entraînement : {outcome.azure_cost_usd:.4f} $")
    status = "modèle courant" if current_version(settings) == version else "non promu (utiliser --promote)"
    print(f"Statut : {status}")
    print(f"Rapport : {report_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
