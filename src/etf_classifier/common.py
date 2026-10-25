"""Small helpers shared by several modules: UTC timestamps, Markdown tables, append-only CSV files."""

import csv
import io
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_timestamp() -> str:
    """Current UTC time, e.g. 2026-10-25T14:03:11Z (the format of every journal and log)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_record_id() -> str:
    """Readable unique identifier without a shared counter, e.g. 20261025-142233-a1b2."""
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def markdown_cell(value) -> str:
    """Text that is safe inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(markdown_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def append_csv_rows(path: str | Path, columns: list[str], rows: list[dict], separator: str = ";") -> None:
    """Appends complete lines in a single write (header first if the file is new); None becomes empty.

    Callers that share the file with other processes hold their own lock around this call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter=separator, lineterminator="\n",
                            extrasaction="ignore")
    if not path.exists() or path.stat().st_size == 0:
        writer.writeheader()
    writer.writerows({column: ("" if row.get(column) is None else row[column]) for column in columns}
                     for row in rows)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def read_csv_table(path: str | Path, columns: list[str], separator: str = ";") -> pd.DataFrame:
    """Whole file as text columns (empty table if the file is missing); missing columns are added empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    table = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False, encoding="utf-8")
    return table.reindex(columns=columns, fill_value="")
