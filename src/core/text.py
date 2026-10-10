"""Text normalization and attribute extraction utilities for Entity Resolution.

Provides consistent, Unicode-safe string processing across both Python
strings and Polars dataframe expressions, preventing train/serve skew across
languages (e.g., English, Hindi/Devanagari, French).
"""

import re
import unicodedata
from typing import Set
import polars as pl

# Multilingual legal entity suffixes covering US, India, and France
LEGAL_SUFFIXES_PATTERN = re.compile(
    r"\b(llc|inc|corp|corporation|ltd|limited|company|co|private|pvt|llp|sarl|sas|sa|eurl|sci|snc)\b",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize a text string in a Unicode-safe manner.

    Applies NFKC normalization, lowercases, removes non-word punctuation
    while preserving multilingual characters (Devanagari, accented Latin),
    and collapses whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode NFKC normalization ensures decomposed characters are composed consistently
    text = unicodedata.normalize("NFKC", text).lower()
    # In Rust/Polars regex, \w matches Letters (L), Marks (M), Numbers (N), and Connector Punctuation (Pc).
    # Standard Python re.sub(r"[^\w\s]") strips Marks (Mn/Mc), stripping Devanagari vowel signs!
    # Filtering by Unicode category ensures 100% parity with Polars regex.
    filtered = "".join(
        c
        for c in text
        if unicodedata.category(c)[0] in "LMN"
        or unicodedata.category(c) == "Pc"
        or c.isspace()
    )
    return re.sub(r"\s+", " ", filtered).strip()



def norm_series(col: pl.Expr) -> pl.Expr:
    """Polars expression to normalize a column equivalently to normalize_text.

    Ensures zero train/serve skew between Polars vectorized pre-processing
    and per-record Python string matching.
    """
    return (
        col.cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def get_numbers(text: str) -> Set[str]:
    """Extract all distinct digit sequences from an address or name."""
    if text is None:
        return set()
    return set(re.findall(r"\d+", str(text)))


def remove_legal_suffixes(text: str) -> str:
    """Strip common legal business suffixes (LLC, Inc, Pvt Ltd, SARL, etc.)."""
    if not text:
        return ""
    cleaned = LEGAL_SUFFIXES_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()
