"""Contrôles avant d'enregistrer une validation : aucune ligne douteuse n'entre dans le journal."""

from etf_classifier.data.parsing import ParsingError, parse_composition


class ValidationError(ValueError):
    """Validation refusée ; `errors` liste toutes les raisons."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(" ; ".join(errors))


def check_validation(name: str, composition_raw: str, categories: list[str], validated_by: str,
                     known_categories: set[str], existing_names: set[str],
                     create_categories: bool, allow_duplicate_name: bool) -> list[str]:
    """Renvoie la liste des problèmes (vide si la validation peut être enregistrée)."""
    errors = []
    if not name.strip():
        errors.append("Le nom de l'ETF est vide.")
    if not validated_by.strip():
        errors.append("L'identité du validateur (nom ou e-mail) est obligatoire.")
    try:
        if any(weight < 0 for weight in parse_composition(composition_raw).values()):
            errors.append("Composition : poids négatif.")
    except ParsingError as exc:
        errors.append(f"{exc}")
    if not categories:
        errors.append("Au moins une catégorie validée est obligatoire.")
    unknown = [c for c in categories if c not in known_categories]
    if unknown and not create_categories:
        errors.append(f"Catégorie(s) inconnue(s) : {', '.join(unknown)}. "
                      "Confirmer leur création (create_categories=True / --create-categories).")
    if name.strip() in existing_names and not allow_duplicate_name:
        errors.append(f"Un ETF nommé « {name.strip()} » est déjà présent dans le jeu de référence. "
                      "Confirmer qu'il s'agit d'un ETF différent (allow_duplicate_name=True).")
    return errors
