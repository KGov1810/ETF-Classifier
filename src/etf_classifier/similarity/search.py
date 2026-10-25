"""ETF déjà catégorisés les plus proches de nouveaux ETF, avec explication et alerte quasi-doublon.

Usage :
    python -m etf_classifier.similarity.search --csv nouveaux_etfs.csv [--model current] [--k 5]
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

EXIT_OK, EXIT_CANCELLED, EXIT_NO_MODEL, EXIT_NO_AZURE, EXIT_NO_INDEX = 0, 1, 2, 3, 4


def _summary(names: list[str], results, version: str) -> str:
    lines = [f"# ETF les plus proches (modèle {version})", ""]
    for name, result in zip(names, results):
        lines += [f"## {name}", ""]
        if result.near_duplicate:
            dup = result.near_duplicate
            lines += [f"> ⚠️ Quasi-doublon de **{dup.name}** ({dup.overlap_pct:.0f} % du poids en commun), "
                      f"catégories : {', '.join(dup.categories)}", ""]
        if result.text_only:
            lines += ["Composition absente : rapprochement sur le texte seul.", ""]
        lines += ["| Rang | ETF | Catégories | Score | Poids en commun | Tickers partagés |", "|---|---|---|---|---|---|"]
        for rank, n in enumerate(result.neighbors, start=1):
            shared = ", ".join(t for t, _ in n.shared_holdings) or "–"
            lines.append(f"| {rank} | {n.name} | {', '.join(n.categories)} | {n.score:.2f} | "
                         f"{n.overlap_pct:.0f} % | {shared} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="ETF les plus proches de nouveaux ETF")
    parser.add_argument("--csv", required=True, help="CSV des nouveaux ETF (Name;Description;Composition)")
    parser.add_argument("--model", default="current", help="Version du modèle (défaut : modèle courant)")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    parser.add_argument("--k", type=int, help="Nombre de voisins (défaut : settings.yaml)")
    parser.add_argument("--output", help="CSV de sortie (un résumé .md est écrit à côté)")
    parser.add_argument("--yes", action="store_true", help="Ne pas demander de confirmation avant la dépense Azure")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")

    settings = load_settings(args.config)
    try:
        bundle = load_bundle(args.model, settings)
    except NoModelError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return EXIT_NO_MODEL
    if bundle.similarity is None:
        print(f"Erreur : le modèle {bundle.version} n'a pas d'index de similarité : le réentraîner "
              "avec `python -m etf_classifier.model.train`.", file=sys.stderr)
        return EXIT_NO_INDEX

    data = load_dataset(args.csv, settings, require_categories=False).data.reset_index(drop=True)
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

    k = args.k or settings["similarity"]["n_neighbors"]
    results = bundle.similarity.query(data, bundle.pipeline, k=k)
    rows = []
    for name, result in zip(data["name"], results):
        duplicate = result.near_duplicate.name if result.near_duplicate else ""
        for rank, n in enumerate(result.neighbors, start=1):
            rows.append({
                "Name": name, "Rank": rank, "NeighborName": n.name, "NeighborCategories": str(n.categories),
                "Score": round(n.score, 4), "CompositionSimilarity": round(n.composition_similarity, 4),
                "TextSimilarity": round(n.text_similarity, 4), "OverlapPct": round(n.overlap_pct, 1),
                "SharedHoldings": str([t for t, _ in n.shared_holdings]), "NearDuplicate": duplicate,
            })
    if args.output:
        output = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = resolve_path(settings["model"]["reports_dir"]) / f"neighbors_{bundle.version}_{stamp}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep=";", index=False, encoding="utf-8")
    output.with_suffix(".md").write_text(_summary(list(data["name"]), results, bundle.version), encoding="utf-8")

    n_duplicates = sum(1 for r in results if r.near_duplicate)
    print(f"{len(data)} ETF traités · {k} voisins chacun · {n_duplicates} quasi-doublon(s) signalé(s)")
    print(f"Voisins : {output}\nRésumé : {output.with_suffix('.md')}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
