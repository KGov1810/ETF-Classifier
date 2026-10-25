"""Versioned storage of the models and registry of the current model.

Each model lives in artifacts/models/<version>/ with its own feature pipeline: the application
and the daily job load a single folder, with no risk of mixing versions.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib

from etf_classifier.config import resolve_path
from etf_classifier.similarity.index import SimilarityIndex

BUNDLE_FILE = "bundle.joblib"
SIMILARITY_FILE = "similarity.joblib"
REGISTRY_FILE = "registry.json"


class NoModelError(LookupError):
    """No model registered, or unknown version."""


@dataclass
class ModelBundle:
    version: str
    path: Path
    model: object
    pipeline: object
    metrics: dict
    similarity: object | None = None  # similarity index (missing from older models)


def _root(settings: dict) -> Path:
    return resolve_path(settings["model"]["artifacts_dir"])


def _read_registry(settings: dict) -> dict:
    path = _root(settings) / REGISTRY_FILE
    if not path.exists():
        return {"current": None, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(settings: dict, registry: dict) -> None:
    path = _root(settings) / REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)  # atomic replacement: never a half-written registry


def _new_version(settings: dict) -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    version, suffix = base, 1
    while (_root(settings) / version).exists():
        suffix += 1
        version = f"{base}-{suffix}"
    return version


def current_version(settings: dict) -> str | None:
    return _read_registry(settings)["current"]


def save_bundle(model, pipeline, metrics: dict, settings: dict, model_card: str,
                version: str | None = None, similarity=None, make_current_if_first: bool = True) -> Path:
    """Writes model + pipeline + metrics.json + MODEL_CARD.md; the first model becomes current
    (unless make_current_if_first=False, used by the retraining dry run)."""
    version = version or _new_version(settings)
    path = _root(settings) / version
    path.mkdir(parents=True, exist_ok=False)
    joblib.dump({"model": model, "pipeline": pipeline}, path / BUNDLE_FILE)
    if similarity is not None:
        similarity.save(path / SIMILARITY_FILE)
    (path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    (path / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    registry = _read_registry(settings)
    registry["history"].append({
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_micro": metrics.get("f1_micro"),
    })
    if registry["current"] is None and make_current_if_first:
        registry["current"] = version
    _write_registry(settings, registry)
    return path


def promote(version: str, settings: dict) -> None:
    """Makes an existing version the current model."""
    if not (_root(settings) / version / BUNDLE_FILE).exists():
        raise NoModelError(f"Unknown model version: {version}")
    registry = _read_registry(settings)
    registry["current"] = version
    _write_registry(settings, registry)


def load_bundle(version: str, settings: dict, embedder=None) -> ModelBundle:
    """Loads a version (or "current") and reattaches the embedder if the pipeline uses one."""
    if version == "current":
        version = current_version(settings)
        if version is None:
            raise NoModelError("No current model: train one with `python -m etf_classifier.model.train`")
    path = _root(settings) / version
    if not (path / BUNDLE_FILE).exists():
        raise NoModelError(f"Unknown model version: {version}")
    content = joblib.load(path / BUNDLE_FILE)
    content["pipeline"].attach_embedder(embedder)
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    similarity = SimilarityIndex.load(path / SIMILARITY_FILE) if (path / SIMILARITY_FILE).exists() else None
    return ModelBundle(version, path, content["model"], content["pipeline"], metrics, similarity)
