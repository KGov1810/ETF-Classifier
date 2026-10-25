"""Lecture de la configuration du projet (config/settings.yaml)."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None) -> dict:
    """Charge le fichier YAML de configuration (par défaut config/settings.yaml)."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    with settings_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    """Rend absolu un chemin de la configuration, relatif à la racine du projet."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
