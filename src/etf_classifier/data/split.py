"""Train/test split by near-duplicate group.

A group is never split in two: otherwise the model would see, in the test set, ETFs that are
almost identical to training ones, and its score would be misleadingly high.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SplitSummary:
    """Summary of the split, shown in the quality report."""

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
    """Splits the dataset into train/test without splitting any group (group_id column required).

    1. Every category present in at least 2 groups gets one group on the test side.
    2. The test set is completed with randomly drawn groups up to the target size,
       without taking away a category's last group on the training side.
    """
    if "group_id" not in data.columns:
        raise ValueError("group_id column missing: call add_groups() before make_split().")

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
    train_count = {c: len(gs) for c, gs in category_groups.items()}  # groups left on the training side

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
