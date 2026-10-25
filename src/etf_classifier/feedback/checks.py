"""Checks before recording a validation: no dubious line enters the journal."""

from etf_classifier.data.parsing import ParsingError, parse_composition


class ValidationError(ValueError):
    """Validation refused; `errors` lists every reason."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def check_validation(name: str, composition_raw: str, categories: list[str], validated_by: str,
                     known_categories: set[str], existing_names: set[str],
                     create_categories: bool, allow_duplicate_name: bool) -> list[str]:
    """Returns the list of problems (empty if the validation can be recorded)."""
    errors = []
    if not name.strip():
        errors.append("The ETF name is empty.")
    if not validated_by.strip():
        errors.append("The validator's identity (name or e-mail) is required.")
    try:
        if any(weight < 0 for weight in parse_composition(composition_raw).values()):
            errors.append("Composition: negative weight.")
    except ParsingError as exc:
        errors.append(f"{exc}")
    if not categories:
        errors.append("At least one validated category is required.")
    unknown = [c for c in categories if c not in known_categories]
    if unknown and not create_categories:
        errors.append(f"Unknown category(ies): {', '.join(unknown)}. "
                      "Confirm their creation (create_categories=True / --create-categories).")
    if name.strip() in existing_names and not allow_duplicate_name:
        errors.append(f"An ETF named '{name.strip()}' is already present in the reference dataset. "
                      "Confirm that it is a different ETF (allow_duplicate_name=True).")
    return errors
