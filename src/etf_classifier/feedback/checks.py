"""Checks before recording a validation: no dubious line enters the journal."""

from etf_classifier.data.parsing import ParsingError, parse_composition


class ValidationError(ValueError):
    """Validation refused; `errors` lists every reason."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def check_validation(ticker: str, name: str, description: str, composition_raw: str, category: str,
                     validated_by: str, known_categories: set[str], create_category: bool) -> list[str]:
    """Returns the list of problems (empty if the validation can be recorded).

    Same rules as a CSV row: ticker and name required, a description or a composition, one category.
    """
    errors = []
    if not ticker:
        errors.append("The ETF ticker is empty.")
    if not name:
        errors.append("The ETF name is empty.")
    if not validated_by.strip():
        errors.append("The validator's identity (name or e-mail) is required.")
    try:
        composition = parse_composition(composition_raw)
        if any(weight < 0 for weight in composition.values()):
            errors.append("Composition: negative weight.")
        if not composition and not description:
            errors.append("A description or a composition is required.")
    except ParsingError as exc:
        errors.append(f"{exc}")
    if not category:
        errors.append("A validated category is required.")
    elif category not in known_categories and not create_category:
        errors.append(f"Unknown category: {category}. "
                      "Confirm its creation (create_category=True / --create-categories).")
    return errors
