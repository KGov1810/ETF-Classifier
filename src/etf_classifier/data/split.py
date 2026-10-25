"""Découpage entraînement/test par groupe de quasi-doublons.

Un groupe n'est jamais coupé en deux : sinon le modèle verrait au test des ETF quasi identiques
à ceux de l'entraînement, et son score serait trompeusement élevé.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SplitSummary:
    """Bilan du découpage, affiché dans le rapport de qualité."""

    test_share: float
    n_train: int
    n_test: int
    rare_categories: dict[str, int] = field(default_factory=dict)
    missing_in_test: list[str] = field(default_factory=list)
    missing_in_train: list[str] = field(default_factory=list)


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    summary: SplitSummary


def make_split(data: pd.DataFrame, test_size: float = 0.2, seed: int = 42, rare_min: int = 5) -> SplitResult:
    """Découpe le jeu en entraînement/test sans couper de groupe (colonne group_id requise).

    1. Chaque catégorie présente dans au moins 2 groupes reçoit un groupe côté test.
    2. Le test est complété par des groupes tirés au hasard jusqu'à la taille visée,
       sans retirer à une catégorie son dernier groupe côté entraînement.
    """
    if "group_id" not in data.columns:
        raise ValueError("Colonne group_id absente : appeler add_groups() avant make_split().")

    rng = np.random.default_rng(seed)
    positions = data.groupby("group_id", sort=True).indices
    order = [g for g in rng.permutation(sorted(positions))]
    rank = {g: r for r, g in enumerate(order)}
    sizes = {g: len(pos) for g, pos in positions.items()}
    group_categories = {
        g: {c for cats in data["categories"].iloc[pos] for c in cats} for g, pos in positions.items()
    }
    category_groups: dict[str, list] = defaultdict(list)
    for g in order:
        for category in group_categories[g]:
            category_groups[category].append(g)

    target = round(test_size * len(data))
    test_groups: set = set()
    train_count = {c: len(gs) for c, gs in category_groups.items()}  # groupes restant côté entraînement

    def move_to_test(group) -> None:
        test_groups.add(group)
        for category in group_categories[group]:
            train_count[category] -= 1

    def keeps_train_coverage(group) -> bool:
        return all(train_count[c] > 1 or len(category_groups[c]) == 1 for c in group_categories[group])

    shared = [c for c, gs in category_groups.items() if len(gs) >= 2]
    for category in sorted(shared, key=lambda c: (len(category_groups[c]), c)):
        if any(g in test_groups for g in category_groups[category]):
            continue
        candidates = [g for g in category_groups[category] if keeps_train_coverage(g)]
        if candidates:
            move_to_test(min(candidates, key=lambda g: (sizes[g], rank[g])))

    test_count = sum(sizes[g] for g in test_groups)
    for group in order:
        if group in test_groups or test_count + sizes[group] > target:
            continue
        if keeps_train_coverage(group) and all(len(category_groups[c]) > 1 for c in group_categories[group]):
            move_to_test(group)
            test_count += sizes[group]

    is_test = data["group_id"].isin(test_groups).to_numpy()
    train, test = data[~is_test].copy(), data[is_test].copy()

    counts = pd.Series([c for cats in data["categories"] for c in cats]).value_counts()
    test_categories = {c for cats in test["categories"] for c in cats}
    train_categories = {c for cats in train["categories"] for c in cats}
    summary = SplitSummary(
        test_share=len(test) / len(data) if len(data) else 0.0,
        n_train=len(train),
        n_test=len(test),
        rare_categories={c: int(n) for c, n in counts.items() if n < rare_min},
        missing_in_test=sorted(set(counts.index) - test_categories),
        missing_in_train=sorted(set(counts.index) - train_categories),
    )
    return SplitResult(train=train, test=test, summary=summary)
