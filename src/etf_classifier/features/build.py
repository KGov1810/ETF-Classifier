"""Construction des variables : estimation du coût, confirmation, apprentissage et sauvegarde du pipeline.

Usage :
    python -m etf_classifier.features.build --csv data/raw/etfs.csv --estimate-only
    python -m etf_classifier.features.build --csv data/raw/etfs.csv          # demande confirmation
    python -m etf_classifier.features.build --csv data/raw/etfs.csv --yes    # sans confirmation (job quotidien)
"""

import argparse
import logging
import sys
from typing import Callable

from etf_classifier.config import load_settings
from etf_classifier.data.loader import prepare_dataset
from etf_classifier.features.embeddings import EmbeddingPlan, EmbeddingsBlock, create_embedder
from etf_classifier.features.pipeline import FeaturePipeline

EXIT_OK, EXIT_CANCELLED = 0, 1


def _density(matrix) -> float:
    cells = matrix.shape[0] * matrix.shape[1]
    return matrix.nnz / cells if cells else 0.0


def announce_embedding_plan(settings: dict, data, embedder) -> tuple[EmbeddingPlan, str]:
    """Affiche ce que coûterait le calcul des embeddings (cache déduit), sans appeler Azure."""
    texts = EmbeddingsBlock(None, embedder.dimensions,
                            settings["features"]["embeddings"]["include_top_tickers"]).texts(data)
    plan = embedder.plan(texts)
    cost = f"{plan.estimate.cost_usd:.4f} $" if plan.estimate else "inconnu (déploiement absent de la grille)"
    approx = " (approximatif)" if plan.estimate and plan.estimate.approximate else ""
    tokens = plan.estimate.input_tokens if plan.estimate else "?"
    print(f"Embeddings : {plan.n_to_compute} texte(s) à calculer, {plan.n_cached} déjà en cache · "
          f"{tokens} tokens · coût estimé {cost}{approx}")
    return plan, cost


def confirm_spending(plan: EmbeddingPlan, cost: str, assume_yes: bool, input_fn: Callable[[str], str]) -> bool:
    """Demande confirmation avant une dépense Azure (sauf si rien à calculer ou assume_yes)."""
    if not plan.n_to_compute or assume_yes:
        return True
    answer = input_fn(f"Confirmer la dépense estimée de {cost} ? [o/N] ").strip().lower()
    if answer in {"o", "oui", "y", "yes"}:
        return True
    print("Annulé : aucun appel Azure effectué.")
    return False


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Construction des variables des ETF")
    parser.add_argument("--csv", required=True, help="Chemin du CSV d'ETF")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    parser.add_argument("--estimate-only", action="store_true", help="Afficher le coût estimé sans rien calculer")
    parser.add_argument("--yes", action="store_true", help="Ne pas demander de confirmation avant la dépense")
    parser.add_argument("--no-azure", action="store_true", help="Construire uniquement les blocs locaux")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")

    settings = load_settings(args.config)
    prepared = prepare_dataset(args.csv, settings)
    embeddings_enabled = settings["features"]["blocks"]["embeddings"]["enabled"] and not args.no_azure
    embedder = create_embedder(settings, client=client) if embeddings_enabled else None
    print(f"{len(prepared.data)} ETF exploitables ({len(prepared.split.train)} en entraînement, "
          f"{len(prepared.split.test)} en test).")

    if embedder is None:
        print("Embeddings : bloc inactif (Azure non configuré ou désactivé).")
    else:
        plan, cost = announce_embedding_plan(settings, prepared.data, embedder)
        if args.estimate_only:
            return EXIT_OK
        if not confirm_spending(plan, cost, args.yes, input_fn):
            return EXIT_CANCELLED
    if args.estimate_only:
        return EXIT_OK

    pipeline = FeaturePipeline(settings, embedder=embedder).fit(prepared.split.train)
    train_features = pipeline.transform(prepared.split.train)
    pipeline.transform(prepared.split.test)
    path = pipeline.save()

    print("\nBlocs construits :")
    for name, matrix in train_features.blocks.items():
        print(f"  - {name} : {matrix.shape[1]} variables, densité {_density(matrix):.1%}")
    print(f"Catégories : {len(pipeline.labels.classes)}")
    if embedder is not None:
        print(f"Coût réel de ce lancement : {embedder.run_cost_usd:.4f} $ ({embedder.n_api_calls} appel(s) Azure)")
    print(f"Pipeline sauvegardé dans {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
