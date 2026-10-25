"""Rapport de qualité du jeu de données (ligne de commande + fichier Markdown).

Usage :
    python -m etf_classifier.data.report --csv data/raw/etfs.csv
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import MissingColumnsError, prepare_dataset
from etf_classifier.data.normalize import find_similar_tickers, normalize_ticker
from etf_classifier.data.validation import RULE_LABELS

MAX_DETAIL_ROWS = 50
EXIT_OK, EXIT_FILE_NOT_FOUND, EXIT_MISSING_COLUMNS = 0, 2, 3


@dataclass
class Report:
    markdown: str
    stats: dict


def _cell(value) -> str:
    """Texte sûr pour une cellule de tableau Markdown."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _issues_section(title: str, issues: list, empty_text: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not issues:
        return lines + [empty_text, ""]
    counts = Counter(issue.rule for issue in issues)
    lines += [_table(["Règle", "Nombre"], [[RULE_LABELS[r], n] for r, n in counts.most_common()]), ""]
    def where(issue) -> str:
        line = issue.line if issue.line is not None else "?"
        return f"journal {line}" if issue.source == "validation" else str(line)

    detail = [[where(i), RULE_LABELS[i.rule], i.message, i.extract]
              for i in sorted(issues, key=lambda i: (i.line or 0))[:MAX_DETAIL_ROWS]]
    lines += ["Détail (ligne du fichier, en-tête = ligne 1) :", "",
              _table(["Ligne", "Règle", "Message", "Extrait"], detail), ""]
    if len(issues) > MAX_DETAIL_ROWS:
        lines += [f"… et {len(issues) - MAX_DETAIL_ROWS} autres.", ""]
    return lines


def build_report(csv_path: str | Path, settings: dict) -> Report:
    """Charge le CSV, calcule les statistiques et produit le rapport Markdown."""
    prepared = prepare_dataset(csv_path, settings)
    data, log, dedup, split = prepared.data, prepared.log, prepared.dedup, prepared.split

    category_counts = pd.Series([c for cats in data["categories"] for c in cats], dtype=object).value_counts()
    raw_tickers = Counter(t for comp in data["composition_raw"] for t in comp)
    tickers = Counter(t for comp in data["composition"] for t in comp)
    similar = find_similar_tickers(normalize_ticker(t) for t in raw_tickers)
    group_sizes = data.groupby("group_id").size().sort_values(ascending=False)

    stats = {
        "total_rows": prepared.total_rows,
        "valid_rows": len(data),
        "excluded_rows": len({(i.row_id, i.line) for i in log.exclusions()}),
        "rows_with_warnings": len({i.row_id for i in log.warnings()}),
        "n_categories": int(category_counts.size),
        "mean_categories_per_etf": float(data["categories"].map(len).mean()) if len(data) else 0.0,
        "n_raw_tickers": len(raw_tickers),
        "n_tickers": len(tickers),
        "n_groups": dedup.n_groups,
        "largest_group_size": dedup.largest_group_size,
    }

    lines = [
        "# Rapport de qualité des données",
        "",
        f"Fichier : `{Path(csv_path).name}` · généré le {datetime.now():%d/%m/%Y %H:%M}",
        "",
        "## Résumé",
        "",
        _table(["Indicateur", "Valeur"], [
            ["Lignes lues", stats["total_rows"]],
            ["Lignes exploitables", stats["valid_rows"]],
            ["Lignes exclues", stats["excluded_rows"]],
            ["Lignes avec avertissement", stats["rows_with_warnings"]],
        ]),
        "",
    ]
    lines += _issues_section("Exclusions", log.exclusions(), "Aucune ligne exclue.")
    lines += _issues_section("Avertissements", log.warnings(), "Aucun avertissement.")

    lines += ["## Catégories", "",
              f"{stats['n_categories']} catégories distinctes, "
              f"{stats['mean_categories_per_etf']:.2f} catégories par ETF en moyenne.", ""]
    if category_counts.size:
        lines += [_table(["Catégorie", "Nombre d'ETF"], [[c, int(n)] for c, n in category_counts.items()]), ""]

    coverage = data["coverage"] if len(data) else pd.Series([0.0])
    lines += ["## Couverture des compositions", "",
              _table(["Indicateur", "Valeur"], [
                  ["Couverture minimale (somme des poids)", f"{coverage.min():.1f}"],
                  ["Couverture médiane", f"{coverage.median():.1f}"],
                  ["Couverture maximale", f"{coverage.max():.1f}"],
                  ["Compositions vides", int((data["n_holdings"] == 0).sum()) if len(data) else 0],
                  ["Nombre médian de lignes", f"{data['n_holdings'].median():.0f}" if len(data) else "0"],
              ]), ""]

    top_n = settings["report"]["top_holdings"]
    lines += ["## Tickers les plus fréquents", "",
              f"{stats['n_raw_tickers']} tickers distincts avant normalisation, "
              f"{stats['n_tickers']} après.", ""]
    if tickers:
        lines += [_table(["Ticker", "Nombre d'ETF"], [[t, n] for t, n in tickers.most_common(top_n)]), ""]

    lines += ["## Tickers ressemblants", "",
              "Même racine, suffixe différent : à vérifier, et à ajouter dans "
              "`config/holding_aliases.csv` s'il s'agit du même titre.", ""]
    if similar:
        rows = [[root, ", ".join(variants)] for root, variants in list(similar.items())[:MAX_DETAIL_ROWS]]
        lines += [_table(["Racine", "Variantes"], rows), ""]
    else:
        lines += ["Aucun.", ""]

    lines += ["## Quasi-doublons", "",
              _table(["Indicateur", "Valeur"], [
                  ["Groupes", dedup.n_groups],
                  ["Liens détectés", dedup.n_links],
                  ["Taille du plus gros groupe", dedup.largest_group_size],
              ]), ""]
    lines += [f"> ⚠️ {w}" for w in dedup.warnings] + ([""] if dedup.warnings else [])
    multi = group_sizes[group_sizes > 1].head(10)
    if len(multi):
        rows = [[g, int(n), ", ".join(data.loc[data["group_id"] == g, "name"].head(8))] for g, n in multi.items()]
        lines += ["Plus gros groupes :", "", _table(["Groupe", "Taille", "ETF (8 premiers)"], rows), ""]

    summary = split.summary
    lines += ["## Découpage entraînement/test", "",
              _table(["Indicateur", "Valeur"], [
                  ["ETF en entraînement", summary.n_train],
                  ["ETF en test", summary.n_test],
                  ["Part du test", f"{summary.test_share:.1%}"],
                  ["Catégories absentes du test", ", ".join(summary.missing_in_test) or "aucune"],
                  ["Catégories absentes de l'entraînement", ", ".join(summary.missing_in_train) or "aucune"],
              ]), "",
              f"Catégories rares (moins de {settings['split']['rare_category_min']} ETF) :", ""]
    if summary.rare_categories:
        lines += [_table(["Catégorie", "Nombre d'ETF"], [[c, n] for c, n in summary.rare_categories.items()]), ""]
    else:
        lines += ["Aucune.", ""]

    return Report(markdown="\n".join(lines), stats=stats)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée en ligne de commande ; renvoie un code de sortie."""
    parser = argparse.ArgumentParser(description="Rapport de qualité du CSV d'ETF")
    parser.add_argument("--csv", required=True, help="Chemin du CSV d'ETF")
    parser.add_argument("--config", help="Fichier de configuration (défaut : config/settings.yaml)")
    parser.add_argument("--output", help="Fichier Markdown de sortie (défaut : reports/data_quality.md)")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Erreur : fichier introuvable : {csv_path}", file=sys.stderr)
        return EXIT_FILE_NOT_FOUND
    try:
        report = build_report(csv_path, settings)
    except MissingColumnsError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return EXIT_MISSING_COLUMNS

    output = Path(args.output) if args.output else resolve_path(settings["report"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"\nRapport écrit dans {output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
