"""Métriques multi-étiquettes, identiques pour la référence triviale et pour le classifieur."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


@dataclass
class Metrics:
    f1_micro: float
    f1_macro: float
    f1_samples: float
    subset_accuracy: float
    hamming_loss: float
    per_category: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "f1_samples": self.f1_samples,
            "subset_accuracy": self.subset_accuracy,
            "hamming_loss": self.hamming_loss,
            "per_category": self.per_category.to_dict(orient="records"),
        }


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(version: str, metrics: Metrics, baseline: Metrics, top: dict, thresholds: dict[str, float],
                  unknown_categories: dict[str, int], info: dict) -> str:
    """Rapport d'évaluation Markdown d'un modèle (jeu de test, évalué une seule fois)."""
    rows = [["F1 macro (métrique principale)", f"{metrics.f1_macro:.3f}", f"{baseline.f1_macro:.3f}"],
            ["F1 micro", f"{metrics.f1_micro:.3f}", f"{baseline.f1_micro:.3f}"],
            ["F1 par ETF", f"{metrics.f1_samples:.3f}", f"{baseline.f1_samples:.3f}"],
            ["Exactitude stricte", f"{metrics.subset_accuracy:.3f}", f"{baseline.subset_accuracy:.3f}"],
            ["Hamming loss (plus bas = mieux)", f"{metrics.hamming_loss:.3f}", f"{baseline.hamming_loss:.3f}"]]
    lines = [
        f"# Rapport d'évaluation : modèle {version}", "",
        _table(["Information", "Valeur"], [[k, v] for k, v in info.items()]), "",
        "## Comparaison à la référence triviale", "",
        "La référence triviale prédit les catégories les plus fréquentes, sans regarder l'ETF.", "",
        _table(["Métrique (jeu de test)", "Modèle", "Référence triviale"], rows), "",
        "## Détail par catégorie", "",
    ]
    details = metrics.per_category.sort_values("support", ascending=False)
    lines += [_table(["Catégorie", "Précision", "Rappel", "F1", "Effectif (test)", "Seuil"],
                     [[r.category, f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", int(r.support),
                       f"{thresholds.get(r.category, 0.5):.2f}"] for r in details.itertuples()]), ""]
    lines += ["## Variables les plus influentes", "",
              "Variables qui poussent le plus vers chaque catégorie (coefficients positifs).", ""]
    lines += [_table(["Catégorie", "Variables"],
                     [[c, ", ".join(name for name, _ in feats) or "aucune (catégorie constante)"]
                      for c, feats in sorted(top.items())]), ""]
    lines += ["## Catégories non apprenables", ""]
    if unknown_categories:
        lines += ["Présentes dans le test mais absentes de l'entraînement : le modèle ne peut pas les prédire.", "",
                  _table(["Catégorie", "ETF concernés (test)"], [[c, n] for c, n in unknown_categories.items()]), ""]
    else:
        lines += ["Aucune.", ""]
    return "\n".join(lines)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> Metrics:
    """Calcule les métriques.

    F1 macro : moyenne sur les catégories présentes dans les vraies valeurs OU dans les prédictions.
    Une catégorie absente des deux côtés n'a rien à mesurer et n'est pas comptée ; une catégorie
    prédite à tort mais absente des vraies valeurs compte (F1 = 0), car c'est une erreur réelle.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    active = np.flatnonzero((y_true.sum(axis=0) + y_pred.sum(axis=0)) > 0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), average=None, zero_division=0
    )
    per_category = pd.DataFrame({
        "category": classes, "precision": precision, "recall": recall, "f1": f1, "support": support,
    })
    if y_true.size == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, per_category)
    return Metrics(
        f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f1_macro=float(f1[active].mean()) if active.size else 0.0,
        f1_samples=float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        subset_accuracy=float((y_true == y_pred).all(axis=1).mean()),
        hamming_loss=float(hamming_loss(y_true, y_pred)),
        per_category=per_category,
    )
