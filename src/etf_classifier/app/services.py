"""Business logic of the interface, without any Streamlit import.

Pages call these functions and only display what they return, so every behaviour
of the app is tested with pytest, without a browser.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from etf_classifier.data.loader import load_records
from etf_classifier.data.validation import RULE_LABELS

TICKER_COLUMN, WEIGHT_COLUMN = "Ticker", "Weight (%)"


@dataclass
class EtfCheck:
    """ETFs checked and normalised like CSV rows; `data` is empty when the input is refused."""

    data: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validators(settings: dict) -> list[str]:
    """Team members allowed to validate (drop-down list of the interface)."""
    return list(settings.get("app", {}).get("validators", []))


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _weight_value(value):
    """Weight as float. Text pasted from Excel ("4.5", "4,5") is converted; anything else is kept
    as text, so the quality rules report it as a non-numeric weight."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return text


def _raw_composition(holdings: pd.DataFrame) -> tuple[str, list[str]]:
    """Editable table → composition in the CSV format "[{'AAPL': 5.0}, ...]"."""
    items, errors = [], []
    for position, row in enumerate(holdings.to_dict("records"), start=1):
        ticker, weight = row.get(TICKER_COLUMN), row.get(WEIGHT_COLUMN)
        if _is_blank(ticker) and _is_blank(weight):
            continue  # empty row of the editable table
        if _is_blank(ticker):
            errors.append(f"Row {position}: ticker missing for weight {weight}.")
            continue
        if _is_blank(weight):
            errors.append(f"Row {position}: weight missing for ticker {str(ticker).strip()}.")
            continue
        items.append({str(ticker).strip().upper(): _weight_value(weight)})
    return str(items), errors


def _describe(issue) -> str:
    """Readable message of a quality issue: rule label, plus the detail when there is one."""
    label = RULE_LABELS[issue.rule]
    return label if issue.message in ("", label) else f"{label} ({issue.message})"


def etf_from_inputs(name: str, description: str, holdings: pd.DataFrame, settings: dict) -> EtfCheck:
    """Checks an ETF typed in the interface with the same rules as a CSV row."""
    errors = []
    if _is_blank(name):
        errors.append("The ETF name is empty.")
    holdings = holdings if holdings is not None else pd.DataFrame(columns=[TICKER_COLUMN, WEIGHT_COLUMN])
    composition, table_errors = _raw_composition(holdings)
    errors += table_errors
    if errors:
        return EtfCheck(pd.DataFrame(), errors)

    cols = settings["csv"]["columns"]
    row = {cols["name"]: str(name).strip(), cols["description"]: str(description or "").strip(),
           cols["composition"]: composition, "__line__": None, "__record__": 0}
    loaded = load_records([row], settings, require_categories=False)
    errors = [_describe(issue) for issue in loaded.log.exclusions()]
    warnings = [_describe(issue) for issue in loaded.log.warnings()]
    data = loaded.data.reset_index(drop=True) if not errors else pd.DataFrame()
    return EtfCheck(data, errors, warnings)


# --- Model and proposals -------------------------------------------------------------------------


class AppError(RuntimeError):
    """Error with a message meant to be shown as is in the interface."""


class CostConfirmationRequired(AppError):
    """The proposal needs Azure embeddings that are not cached yet: the user must confirm the cost."""

    def __init__(self, plan):
        self.plan = plan
        super().__init__(f"Embedding {plan.n_to_compute} new text(s) needs a confirmation of the cost.")


@dataclass
class Proposal:
    """What the interface shows for one ETF."""

    name: str
    description: str
    composition: dict
    categories: list[str]
    scores: list[tuple[str, float]]
    uncertain: bool
    neighbors: list
    near_duplicate: object | None
    text_only: bool
    similarity_available: bool
    model_version: str


def load_current_model(settings: dict, client=None):
    """Current model, with the Azure embedder attached when its pipeline uses embeddings."""
    from etf_classifier.features.embeddings import create_embedder
    from etf_classifier.model.registry import NoModelError, load_bundle

    try:
        bundle = load_bundle("current", settings)
    except NoModelError as exc:
        raise AppError(str(exc)) from exc
    if bundle.pipeline.uses_embeddings:
        embedder = create_embedder(settings, client=client)
        if embedder is None:
            raise AppError(f"Model {bundle.version} uses Azure embeddings, but Azure OpenAI is not configured "
                           "(see .env.example).")
        bundle.pipeline.attach_embedder(embedder)
    return bundle


def _embedder(bundle):
    from etf_classifier.features.embeddings import EmbeddingsBlock

    for block in bundle.pipeline.blocks:
        if isinstance(block, EmbeddingsBlock):
            return block.embedder
    return None


def estimate(data: pd.DataFrame, bundle, settings: dict):
    """Cost of the embeddings still to compute (EmbeddingPlan), or None for a model without embeddings."""
    from etf_classifier.features.build import plan_embeddings

    if not bundle.pipeline.uses_embeddings:
        return None
    return plan_embeddings(settings, data, _embedder(bundle))


def propose(data: pd.DataFrame, bundle, settings: dict, confirmed: bool = False) -> list[Proposal]:
    """Model proposals and closest ETFs. Azure is only called if nothing is to be paid or confirmed=True."""
    plan = estimate(data, bundle, settings)
    if plan is not None and plan.n_to_compute and not confirmed:
        raise CostConfirmationRequired(plan)
    data = data.reset_index(drop=True)
    cfg = settings["model"]
    predictions = bundle.model.predict(bundle.pipeline.transform(data).matrix,
                                       cfg["max_categories"], cfg["uncertain_threshold"])
    if bundle.similarity is not None:
        results = bundle.similarity.query(data, bundle.pipeline, k=settings["similarity"]["n_neighbors"])
    else:
        results = [None] * len(data)
    n_scores = settings.get("app", {}).get("n_scores_shown", 10)
    proposals = []
    for (_, etf), prediction, result in zip(data.iterrows(), predictions, results):
        proposals.append(Proposal(
            name=etf["name"], description=etf["description"], composition=dict(etf["composition_raw"]),
            categories=list(prediction.categories), scores=list(prediction.scores[:n_scores]),
            uncertain=bool(prediction.uncertain),
            neighbors=result.neighbors if result else [], near_duplicate=result.near_duplicate if result else None,
            text_only=result.text_only if result else False, similarity_available=result is not None,
            model_version=bundle.version,
        ))
    return proposals


# --- Validation, cancellation, history -----------------------------------------------------------


def _store(settings: dict):
    from etf_classifier.feedback.store import FeedbackStore

    return FeedbackStore(settings)


def _feedback_call(action):
    """Runs a journal operation and turns its refusals into messages for the interface."""
    from etf_classifier.feedback.checks import ValidationError
    from etf_classifier.feedback.store import FeedbackError

    try:
        return action()
    except ValidationError as exc:
        raise AppError(" ".join(exc.errors)) from exc
    except FeedbackError as exc:
        raise AppError(str(exc)) from exc
    except PermissionError as exc:  # e.g. journal open in Excel on Windows
        raise AppError("The journal is used by another application (Excel?): close it and try again.") from exc


def known_categories(settings: dict) -> list[str]:
    """Categories that can be selected in the validation form, sorted."""
    return sorted(_feedback_call(lambda: _store(settings).known_categories()))


def validate(proposal: Proposal, categories: list[str], validator: str, settings: dict, comment: str = "",
             new_categories: dict[str, str] | None = None) -> dict:
    """Records the validation of a proposal; new_categories = {name: description} of categories to create."""
    if not str(validator or "").strip():
        raise AppError("Choose a validator before validating.")
    new_categories = new_categories or {}
    etf = {"name": proposal.name, "description": proposal.description, "composition": proposal.composition}
    return _feedback_call(lambda: _store(settings).record_validation(
        etf, list(categories) + [c for c in new_categories if c not in categories], validator,
        predicted_categories=proposal.categories, model_version=proposal.model_version, comment=comment,
        create_categories=bool(new_categories), category_descriptions=new_categories))


def cancel(validation_id: str, validator: str, comment: str, settings: dict) -> dict:
    """Cancels a validation (a comment is mandatory); nothing is deleted from the journal."""
    if not str(validator or "").strip():
        raise AppError("Choose a validator before cancelling.")
    if not str(comment or "").strip():
        raise AppError("A comment is required to cancel a validation.")
    return _feedback_call(lambda: _store(settings).cancel_validation(validation_id, validator, comment))


HISTORY_COLUMNS = ["ValidationId", "ValidatedAt", "Name", "Category", "PredictedCategories", "Decision",
                   "ValidatedBy", "ModelVersion", "Comment", "Cancelled", "CancellationComment"]


def history(settings: dict, since=None, validator: str | None = None, decision: str | None = None,
            name: str | None = None) -> pd.DataFrame:
    """Validations, newest first, with cancelled ones flagged (cancellation lines are not listed)."""
    from etf_classifier.feedback.store import CANCELLATION

    store = _store(settings)
    if not store.journal_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    journal = _feedback_call(store.read_journal)
    cancellations = journal[journal["Decision"] == CANCELLATION].set_index("CancelsId")["Comment"]
    result = journal[journal["Decision"] != CANCELLATION].copy()
    result["Cancelled"] = result["ValidationId"].isin(cancellations.index)
    result["CancellationComment"] = result["ValidationId"].map(cancellations).fillna("")
    if since is not None:
        result = result[pd.to_datetime(result["ValidatedAt"]).dt.date >= since]
    if validator:
        result = result[result["ValidatedBy"] == validator]
    if decision:
        result = result[result["Decision"] == decision]
    if name:
        result = result[result["Name"].str.contains(name, case=False, regex=False)]
    result = result.iloc[::-1]  # the journal is in chronological order
    return result[HISTORY_COLUMNS].reset_index(drop=True)


# --- Batch ---------------------------------------------------------------------------------------


@dataclass
class BatchLoad:
    """Usable ETFs of an uploaded CSV, and the rows that were refused (with their reason)."""

    data: pd.DataFrame
    exclusions: list[dict]
    warnings: list[dict]


def load_batch(content: bytes, settings: dict) -> BatchLoad:
    """Reads an uploaded CSV of new ETFs (Name;Description;Composition, Category optional)."""
    import tempfile
    from pathlib import Path

    from etf_classifier.data.loader import MissingColumnsError, read_raw_rows

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "batch.csv"
        path.write_bytes(content)
        try:
            rows, bad_lines = read_raw_rows(path, settings, require_categories=False)
        except MissingColumnsError as exc:
            raise AppError(f"{exc}. Expected: Name;Description;Composition (Category optional).") from exc
        except UnicodeDecodeError as exc:
            raise AppError("The file is not UTF-8 text: save it as 'CSV UTF-8' and upload it again.") from exc
    loaded = load_records(rows, settings, require_categories=False, bad_lines=bad_lines)
    names = {row["__record__"]: row.get(settings["csv"]["columns"]["name"], "") for row in rows}

    def describe(issue) -> dict:
        return {"line": issue.line, "name": names.get(issue.row_id, ""), "reason": _describe(issue)}

    return BatchLoad(loaded.data.reset_index(drop=True),
                     [describe(issue) for issue in loaded.log.exclusions()],
                     [describe(issue) for issue in loaded.log.warnings()])


def queue_order(proposals: list, uncertain_first: bool) -> list[int]:
    """Order in which the queue shows the proposals (positions in the batch)."""
    positions = list(range(len(proposals)))
    if uncertain_first:
        positions.sort(key=lambda i: not proposals[i].uncertain)  # stable: keeps the file order otherwise
    return positions


# --- Monitoring ----------------------------------------------------------------------------------


def monitoring(settings: dict, since=None):
    """(feedback report, Azure cost report) over the period: the same reports as the commands."""
    from etf_classifier.config import resolve_path
    from etf_classifier.costs.pricing import load_pricing
    from etf_classifier.costs.report import build_cost_report
    from etf_classifier.feedback.report import build_feedback_report

    feedback = _feedback_call(lambda: build_feedback_report(settings, since))
    cfg = settings["costs"]
    costs = build_cost_report(resolve_path(cfg["usage_log_path"]), load_pricing(resolve_path(cfg["pricing_path"])),
                              since=since, stale_after_days=cfg["stale_after_days"])
    return feedback, costs


def without_title(markdown: str) -> str:
    """Report body for a page: first-level title removed, section titles made smaller."""
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    lines = ["#### " + line[3:] if line.startswith("## ") else line for line in lines]
    return "\n".join(lines).strip()


def retraining_runs(settings: dict, limit: int = 10) -> pd.DataFrame:
    """Last daily retraining runs, newest first."""
    from etf_classifier.retraining.history import read_history

    history = read_history(settings)
    columns = ["started_at", "outcome", "current_version", "candidate_version", "current_f1_macro",
               "candidate_f1_macro", "azure_cost_usd", "reasons"]
    return history.iloc[::-1].head(limit)[columns].reset_index(drop=True)
