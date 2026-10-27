"""Business logic of the interface, without any Streamlit import.

Pages call these functions and only display what they return, so every behaviour
of the app is tested with pytest, without a browser.
"""

import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from etf_classifier.config import resolve_path
from etf_classifier.costs.pricing import load_pricing
from etf_classifier.costs.report import build_cost_report
from etf_classifier.data.loader import MissingColumnsError, load_records, read_raw_rows
from etf_classifier.data.parsing import format_composition
from etf_classifier.data.validation import RULE_LABELS
from etf_classifier.features.embeddings import EmbeddingsBlock, create_embedder, plan_embeddings
from etf_classifier.feedback.categories import CategoryRegistry
from etf_classifier.feedback.checks import ValidationError
from etf_classifier.feedback.report import build_feedback_report
from etf_classifier.feedback.store import CANCELLATION, FeedbackError, FeedbackStore
from etf_classifier.llm.justify import Justifier, LLMUnavailable
from etf_classifier.llm.prompts import AnswerError, Justification
from etf_classifier.llm.store import JustificationStore
from etf_classifier.model.check import AGREEMENT, DISAGREEMENT, UNKNOWN, check_established
from etf_classifier.model.registry import NoModelError, load_bundle
from etf_classifier.retraining.history import read_history

TICKER_COLUMN, WEIGHT_COLUMN = "Ticker", "Weight (%)"
logger = logging.getLogger(__name__)


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
        items.append((str(ticker).strip().upper(), _weight_value(weight)))
    return format_composition(items), errors


def _describe(issue) -> str:
    """Readable message of a quality issue: rule label, plus the detail when there is one."""
    label = RULE_LABELS[issue.rule]
    return label if issue.message in ("", label) else f"{label} ({issue.message})"


def etf_from_inputs(ticker: str, name: str, description: str, holdings: pd.DataFrame, settings: dict) -> EtfCheck:
    """Checks an ETF typed in the interface with the same rules as a CSV row."""
    errors = []
    if _is_blank(ticker):
        errors.append("The ETF ticker is empty.")
    if _is_blank(name):
        errors.append("The ETF name is empty.")
    holdings = holdings if holdings is not None else pd.DataFrame(columns=[TICKER_COLUMN, WEIGHT_COLUMN])
    composition, table_errors = _raw_composition(holdings)
    errors += table_errors
    if errors:
        return EtfCheck(pd.DataFrame(), errors)

    cols = settings["csv"]["columns"]
    row = {cols["ticker"]: str(ticker).strip(), cols["name"]: str(name).strip(),
           cols["description"]: str(description or "").strip(),
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

    ticker: str
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
    established: str = ""                    # classification given in the uploaded CSV ("" if none)
    check: str | None = None                 # agreement, disagreement, unknown (None without classification)
    established_score: float | None = None   # model score of the established category

    @property
    def category(self) -> str:
        """The proposed category (the model proposes exactly one)."""
        return self.categories[0]


def load_current_model(settings: dict, client=None):
    """Current model, with the Azure embedder attached when its pipeline uses embeddings."""
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
    for block in bundle.pipeline.blocks:
        if isinstance(block, EmbeddingsBlock):
            return block.embedder
    return None


def estimate(data: pd.DataFrame, bundle, settings: dict):
    """Cost of the embeddings still to compute (EmbeddingPlan), or None for a model without embeddings."""
    if not bundle.pipeline.uses_embeddings:
        return None
    return plan_embeddings(settings, data, _embedder(bundle))


def propose(data: pd.DataFrame, bundle, settings: dict, confirmed: bool = False) -> list[Proposal]:
    """Model proposals and closest ETFs. Azure is only called if nothing is to be paid or confirmed=True."""
    plan = estimate(data, bundle, settings)
    if plan is not None and plan.n_to_compute and not confirmed:
        raise CostConfirmationRequired(plan)
    data = data.reset_index(drop=True)
    predictions = bundle.model.predict(bundle.pipeline.transform(data).matrix,
                                       settings["model"]["uncertain_threshold"])
    if bundle.similarity is not None:
        results = bundle.similarity.query(data, bundle.pipeline, k=settings["similarity"]["n_neighbors"])
    else:
        results = [None] * len(data)
    n_scores = settings.get("app", {}).get("n_scores_shown", 10)
    proposals = []
    for (_, etf), prediction, result in zip(data.iterrows(), predictions, results):
        check = check_established(prediction, etf["categories"][0] if etf["categories"] else "",
                                  bundle.model.classes)
        proposals.append(Proposal(
            ticker=etf["ticker"], name=etf["name"], description=etf["description"],
            composition=dict(etf["composition_raw"]),
            categories=list(prediction.categories), scores=list(prediction.scores[:n_scores]),
            uncertain=bool(prediction.uncertain),
            neighbors=result.neighbors if result else [], near_duplicate=result.near_duplicate if result else None,
            text_only=result.text_only if result else False, similarity_available=result is not None,
            model_version=bundle.version, established=check.established, check=check.verdict,
            established_score=check.established_score,
        ))
    return proposals


# --- Validation, cancellation, history -----------------------------------------------------------


def _feedback_call(action):
    """Runs a journal operation and turns its refusals into messages for the interface."""
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
    return sorted(_feedback_call(lambda: FeedbackStore(settings).known_categories()))


def validate(proposal: Proposal, category: str, validator: str, settings: dict, comment: str = "",
             create_category: bool = False, category_description: str = "",
             justification_id: str | None = None) -> dict:
    """Records the validation of a proposal with one category (created if create_category is set).

    justification_id: LLM justification shown before validating, linked to the validation.
    """
    if not str(validator or "").strip():
        raise AppError("Choose a validator before validating.")
    etf = {"ticker": proposal.ticker, "name": proposal.name, "description": proposal.description,
           "composition": proposal.composition}
    record = _feedback_call(lambda: FeedbackStore(settings).record_validation(
        etf, category, validator, predicted_category=proposal.category, model_version=proposal.model_version,
        comment=comment, create_category=create_category, category_description=category_description))
    if justification_id:
        try:
            JustificationStore(settings).link(justification_id, record["ValidationId"])
        except Exception as exc:  # noqa: BLE001 - the validation is saved: report, never hide it behind a crash
            logger.warning("Justification %s not linked to %s: %s", justification_id, record["ValidationId"], exc)
            record = {**record, "warning": f"Validation saved, but the LLM justification could not be linked: {exc}"}
    return record


def cancel(validation_id: str, validator: str, comment: str, settings: dict) -> dict:
    """Cancels a validation (a comment is mandatory); nothing is deleted from the journal."""
    if not str(validator or "").strip():
        raise AppError("Choose a validator before cancelling.")
    if not str(comment or "").strip():
        raise AppError("A comment is required to cancel a validation.")
    return _feedback_call(lambda: FeedbackStore(settings).cancel_validation(validation_id, validator, comment))


HISTORY_COLUMNS = ["ValidationId", "ValidatedAt", "Ticker", "Name", "Category", "PredictedCategory", "Decision",
                   "ValidatedBy", "ModelVersion", "Comment", "Cancelled", "CancellationComment"]


def history(settings: dict, since=None, validator: str | None = None, decision: str | None = None,
            name: str | None = None) -> pd.DataFrame:
    """Validations, newest first, with cancelled ones flagged (cancellation lines are not listed)."""
    store = FeedbackStore(settings)
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
    if name:  # matches the name or the ticker
        result = result[result["Name"].str.contains(name, case=False, regex=False)
                        | result["Ticker"].str.contains(name, case=False, regex=False)]
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
    """Reads an uploaded CSV of ETFs (input format; a Classification, if given, is checked by the model)."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "batch.csv"
        path.write_bytes(content)
        try:
            rows, bad_lines = read_raw_rows(path, settings, require_categories=False)
        except MissingColumnsError as exc:
            cols = settings["csv"]["columns"]
            expected = ";".join(cols[k] for k in ("ticker", "name", "description", "composition"))
            raise AppError(f"{exc}. Expected: {expected} ({cols['category']} optional).") from exc
        except UnicodeDecodeError as exc:
            raise AppError("The file is not UTF-8 text: save it as 'CSV UTF-8' and upload it again.") from exc
    loaded = load_records(rows, settings, require_categories=False, bad_lines=bad_lines)
    names = {row["__record__"]: row.get(settings["csv"]["columns"]["name"], "") for row in rows}

    def describe(issue) -> dict:
        return {"line": issue.line, "name": names.get(issue.row_id, ""), "reason": _describe(issue)}

    return BatchLoad(loaded.data.reset_index(drop=True),
                     [describe(issue) for issue in loaded.log.exclusions()],
                     [describe(issue) for issue in loaded.log.warnings()])


def _priority(proposal) -> int:
    if proposal.check in (DISAGREEMENT, UNKNOWN):
        return 0
    return 1 if proposal.uncertain else 2


def queue_order(proposals: list, priority_first: bool) -> list[int]:
    """Order in which the queue shows the proposals (positions in the batch).

    priority_first: disagreements with the established classification and unknown categories, then uncertain
    proposals, then the rest; the file order is kept within each group.
    """
    positions = list(range(len(proposals)))
    if priority_first:
        positions.sort(key=lambda i: _priority(proposals[i]))
    return positions


def check_counts(proposals: list) -> dict[str, int]:
    """Number of agreements, disagreements and unknown categories among the established classifications."""
    verdicts = [p.check for p in proposals]
    return {verdict: verdicts.count(verdict) for verdict in (AGREEMENT, DISAGREEMENT, UNKNOWN)}


# --- Monitoring ----------------------------------------------------------------------------------


def monitoring(settings: dict, since=None):
    """(feedback report, Azure cost report) over the period: the same reports as the commands."""
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
    history = read_history(settings)
    columns = ["started_at", "outcome", "current_version", "candidate_version", "current_f1_macro",
               "candidate_f1_macro", "azure_cost_usd", "reasons"]
    return history.iloc[::-1].head(limit)[columns].reset_index(drop=True)


# --- LLM justification ---------------------------------------------------------------------------


def category_descriptions(settings: dict) -> dict[str, str]:
    """Known categories with their description (empty for most historical ones)."""
    known = known_categories(settings)  # initialises the registry from the original CSV if needed
    registry = CategoryRegistry(settings).read()
    descriptions = dict(zip(registry["category"], registry["description"]))
    return {name: descriptions.get(name, "") for name in known}


def llm_justifier(settings: dict, client=None):
    return Justifier(settings, client=client)


def _llm_call(action):
    try:
        return action()
    except (LLMUnavailable, AnswerError) as exc:
        raise AppError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - Azure or network failure: shown to the user, never hidden
        raise AppError(f"The call to Azure OpenAI failed: {exc}") from exc


def llm_estimate(justifier, proposal: Proposal, settings: dict):
    """Estimated cost of the justification (and whether it is already cached, hence free)."""
    return _llm_call(lambda: justifier.estimate(proposal, category_descriptions(settings)))


def llm_justify(justifier, proposal: Proposal, settings: dict, confirmed: bool = False):
    """Justification of a proposal; Azure is only called with confirmed=True (or not at all when cached)."""
    return _llm_call(lambda: justifier.justify(proposal, category_descriptions(settings), confirmed=confirmed))


def suggested_category(selection: str, justification) -> str:
    """Category to select after "Apply the suggestion": the LLM's alternative, or the current selection."""
    return justification.alternative or selection


def justification_for_validation(settings: dict, validation_id: str) -> dict | None:
    return JustificationStore(settings).for_validation(validation_id)


def stored_justification(stored: dict):
    """Justification object from a line of data/llm_justifications.jsonl."""
    answer = dict(stored["answer"])
    answer.setdefault("notes", [])
    return Justification(**answer)
