"""Check of a classification already established by someone against the model's proposal.

Pure function, shared by `model.predict` and the batch page of the app, so both give the same verdicts.
"""

from dataclasses import dataclass

AGREEMENT, DISAGREEMENT, UNKNOWN = "agreement", "disagreement", "unknown"


@dataclass
class Check:
    verdict: str | None              # agreement, disagreement, unknown (category the model does not know), or None
    established: str                 # "" when no classification was given
    established_score: float | None  # model score of the established category (None if unknown or not given)


def check_established(prediction, established: str, classes: list[str]) -> Check:
    """Compares the established category with the model's proposal (a Prediction)."""
    established = (established or "").strip()
    if not established:
        return Check(None, "", None)
    if established not in classes:
        return Check(UNKNOWN, established, None)
    score = dict(prediction.scores)[established]
    verdict = AGREEMENT if prediction.categories[0] == established else DISAGREEMENT
    return Check(verdict, established, score)
