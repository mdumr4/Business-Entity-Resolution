"""Unit tests for text normalization and suffix extraction."""

import unittest
import polars as pl
from src.core.text import normalize_text, norm_series, get_numbers, remove_legal_suffixes


class TestTextCore(unittest.TestCase):
    def test_normalize_text_multilingual_devanagari(self):
        # Ensure Devanagari vowel signs are preserved and not stripped as punctuation
        text_hindi = "अमूल डेरी प्राइवेट लिमिटेड"
        normalized = normalize_text(text_hindi)
        self.assertIn("अमूल", normalized)
        self.assertIn("डेरी", normalized)

    def test_normalize_text_latin_and_punctuation(self):
        raw = "  Amazon Web Services, Inc. & Co.!  "
        normalized = normalize_text(raw)
        self.assertEqual(normalized, "amazon web services inc co")

    def test_polars_parity(self):
        raw_list = ["Walmart Inc.", "स्टेट बैंक ऑफ इंडिया", "L'Oréal S.A.R.L."]
        df = pl.DataFrame({"raw": raw_list}).with_columns(norm_series(pl.col("raw")).alias("norm"))
        polars_results = df["norm"].to_list()
        python_results = [normalize_text(x) for x in raw_list]
        self.assertEqual(polars_results, python_results)

    def test_get_numbers(self):
        self.assertEqual(get_numbers("123 Main St, Suite 400"), {"123", "400"})
        self.assertEqual(get_numbers("No digits here"), set())
        self.assertEqual(get_numbers(None), set())

    def test_remove_legal_suffixes(self):
        self.assertEqual(remove_legal_suffixes("tata consultancy services pvt ltd"), "tata consultancy services")
        self.assertEqual(remove_legal_suffixes("apple inc"), "apple")
        self.assertEqual(remove_legal_suffixes("societe generale sarl"), "societe generale")


if __name__ == "__main__":
    unittest.main()
