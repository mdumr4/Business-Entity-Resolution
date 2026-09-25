import re
import polars as pl

# Hand-written normalization dictionaries explicitly allowed by the rules
ADDRESS_REPLACEMENTS = {
    r"\b(street|stree|st)\b": "st",
    r"\b(avenue|ave)\b": "ave",
    r"\b(boulevard|blvd)\b": "blvd",
    r"\b(road|rd)\b": "rd",
    r"\b(drive|dr)\b": "dr",
    r"\b(lane|ln)\b": "ln",
    r"\b(suite|ste)\b": "ste",
    r"\b(apartment|apt)\b": "apt",
    r"\b(highway|hwy)\b": "hwy",
    r"\b(north|n)\b": "n",
    r"\b(south|s)\b": "s",
    r"\b(east|e)\b": "e",
    r"\b(west|w)\b": "w",
}

def normalize_text(series: pl.Series) -> pl.Series:
    """
    Pure-algorithm text normalization. 
    Complies with rules: No external geo-databases, no APIs.
    """
    # 1. Lowercase
    s = series.str.to_lowercase()
    
    # 2. Remove special characters (keep alphanumeric and spaces)
    s = s.str.replace_all(r"[^a-z0-9\s]", " ")
    
    # 3. Apply hand-written dictionary replacements using regex
    for pattern, replacement in ADDRESS_REPLACEMENTS.items():
        s = s.str.replace_all(pattern, replacement)
        
    # 4. Collapse multiple spaces into a single space and strip
    s = s.str.replace_all(r"\s+", " ").str.strip_chars()
    
    return s

def prepare_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Applies normalization to both name and address columns."""
    return df.with_columns(
        normalize_text(pl.col("business_name").fill_null("")).alias("norm_name"),
        normalize_text(pl.col("business_address").fill_null("")).alias("norm_address")
    )
