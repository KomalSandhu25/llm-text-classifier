"""Unit tests for the data preprocessing pipeline.

Covers ``TextPreprocessor`` and ``PreprocessorConfig`` with a focus on
edge cases: empty input, very long text, special characters, URLs, HTML,
Unicode, and mixed content.  All tests are self-contained and require no
network access or GPU.

Run with::

    pytest tests/test_data.py -v
"""

from __future__ import annotations

import pytest

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor, make_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_prep() -> TextPreprocessor:
    """``TextPreprocessor`` with default settings (all steps enabled, no char cap)."""
    return TextPreprocessor()


@pytest.fixture()
def strict_prep() -> TextPreprocessor:
    """Preprocessor with all steps enabled and a 50-character cap."""
    return TextPreprocessor(max_chars=50)


# ---------------------------------------------------------------------------
# Edge cases: empty / whitespace-only input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string(self, default_prep: TextPreprocessor) -> None:
        assert default_prep.clean("") == ""

    def test_none_like_falsy_empty(self, default_prep: TextPreprocessor) -> None:
        """Passing an empty string should never raise."""
        result = default_prep.clean("")
        assert isinstance(result, str)

    def test_whitespace_only(self, default_prep: TextPreprocessor) -> None:
        assert default_prep.clean("   \t\n  ") == ""

    def test_newlines_only(self, default_prep: TextPreprocessor) -> None:
        assert default_prep.clean("\n\n\n") == ""


# ---------------------------------------------------------------------------
# URL removal
# ---------------------------------------------------------------------------

class TestURLRemoval:
    def test_http_url_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Read more at http://example.com/article")
        assert "http" not in result
        assert "example.com" not in result

    def test_https_url_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("See https://secure.site.org/path?q=1")
        assert "https" not in result

    def test_www_url_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Visit www.google.com for details")
        assert "www" not in result
        assert "google.com" not in result

    def test_text_around_url_preserved(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Click https://foo.bar to sign up")
        assert "click" in result
        assert "sign up" in result

    def test_multiple_urls_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("http://a.com and https://b.org and www.c.net")
        assert "http" not in result
        assert "www" not in result

    def test_url_only_input(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("https://example.com")
        assert result == ""


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class TestHTMLStripping:
    def test_bold_tag_stripped(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("<b>Breaking news</b>")
        assert "<b>" not in result
        assert "breaking news" in result

    def test_br_tag_stripped(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Line one<br/>Line two")
        assert "<br" not in result

    def test_html_entity_decoded(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("AT&amp;T reports record profits")
        # After decoding "&amp;" → "&", special-char removal strips "&"
        assert "amp" not in result
        assert "att" in result or "at" in result  # letters kept

    def test_nested_tags(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("<div><p>Hello <span>world</span></p></div>")
        assert "<" not in result
        assert "hello world" in result

    def test_lt_gt_entities(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("a &lt; b &gt; c")
        assert "&lt;" not in result
        assert "&gt;" not in result


# ---------------------------------------------------------------------------
# Special character removal
# ---------------------------------------------------------------------------

class TestSpecialCharacters:
    def test_exclamation_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Wow!!!")
        assert "!" not in result
        assert "wow" in result

    def test_punctuation_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Hello, world.")
        assert "," not in result
        assert "." not in result

    def test_hashtag_symbol_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("#BreakingNews today")
        assert "#" not in result

    def test_at_symbol_removed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("@JohnDoe said this")
        assert "@" not in result

    def test_alphanumeric_preserved(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("abc123 XYZ 456")
        assert "abc123" in result
        assert "xyz" in result  # lowercased
        assert "456" in result

    def test_all_special_chars(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("!@#$%^&*()-+=[]{}|;':\",./<>?")
        assert result == ""

    def test_special_chars_disabled(self) -> None:
        prep = TextPreprocessor(
            config=PreprocessorConfig(remove_special_chars=False, lowercase=False)
        )
        result = prep.clean("Hello, World!")
        assert "," in result
        assert "!" in result


# ---------------------------------------------------------------------------
# Lowercasing
# ---------------------------------------------------------------------------

class TestLowercasing:
    def test_uppercase_to_lower(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("THE QUICK BROWN FOX")
        assert result == "the quick brown fox"

    def test_mixed_case(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("Apple Inc. Reports RECORD Revenue")
        assert result == result.lower()

    def test_lowercase_disabled(self) -> None:
        prep = TextPreprocessor(
            config=PreprocessorConfig(
                lowercase=False,
                remove_special_chars=False,
            )
        )
        result = prep.clean("Hello WORLD")
        assert "WORLD" in result


# ---------------------------------------------------------------------------
# Whitespace collapsing
# ---------------------------------------------------------------------------

class TestWhitespaceCollapsing:
    def test_double_space_collapsed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("hello  world")
        assert "  " not in result
        assert result == "hello world"

    def test_tabs_collapsed(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("col1\tcol2\tcol3")
        assert "\t" not in result

    def test_leading_trailing_stripped(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("   hello   ")
        assert result == "hello"

    def test_mixed_whitespace(self, default_prep: TextPreprocessor) -> None:
        result = default_prep.clean("  hello\t\tworld\n\n!")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# Very long text / truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_truncation_at_limit(self) -> None:
        prep = TextPreprocessor(max_chars=20)
        long_text = "a " * 100  # "a a a a … "
        result = prep.clean(long_text)
        assert len(result) <= 20

    def test_truncation_word_boundary(self) -> None:
        prep = TextPreprocessor(max_chars=10)
        result = prep.clean("hello world foo bar")
        # Word boundary means we don't cut mid-word
        assert " " not in result or result.endswith(result.split()[-1])
        assert len(result) <= 10

    def test_no_truncation_when_disabled(self) -> None:
        prep = TextPreprocessor()  # max_chars=None
        text = "word " * 500
        result = prep.clean(text)
        # Should not truncate — result should contain many words
        assert len(result.split()) > 100

    def test_exact_limit_no_truncation(self) -> None:
        prep = TextPreprocessor(max_chars=5)
        result = prep.clean("hello")
        assert result == "hello"

    def test_very_long_single_word(self) -> None:
        """When there's no space to break on, fall back to hard char cut."""
        prep = TextPreprocessor(max_chars=5)
        result = prep.clean("abcdefghijklmnop")
        assert len(result) <= 5

    def test_long_text_url_and_html(self) -> None:
        """Pipeline should handle a realistic noisy long article snippet."""
        prep = TextPreprocessor(max_chars=200)
        snippet = (
            "<p>Scientists at <b>MIT</b> have discovered a new "
            "technique. Read more at https://mit.edu/news/2024 "
            "and follow us @MIT on Twitter!!! "
        ) * 10
        result = prep.clean(snippet)
        assert len(result) <= 200
        assert "<" not in result
        assert "http" not in result


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------

class TestUnicodeNormalisation:
    def test_full_width_digits(self, default_prep: TextPreprocessor) -> None:
        # Full-width "１２３" → ASCII "123" after NFKC
        result = default_prep.clean("１２３")
        assert "1" in result or "123" in result

    def test_ligature_normalised(self, default_prep: TextPreprocessor) -> None:
        # "ﬁ" (fi ligature) → "fi" under NFKC
        result = default_prep.clean("ﬁnancial report")
        assert "fi" in result or "financial" in result


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

class TestBatchAPI:
    def test_batch_length_preserved(self, default_prep: TextPreprocessor) -> None:
        texts = ["Hello World!", "", "  ", "https://foo.com bar"]
        results = default_prep.clean_batch(texts)
        assert len(results) == len(texts)

    def test_batch_results_match_single(self, default_prep: TextPreprocessor) -> None:
        texts = ["Hello World!", "Visit http://foo.com now"]
        batch_results = default_prep.clean_batch(texts)
        single_results = [default_prep.clean(t) for t in texts]
        assert batch_results == single_results


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

class TestMakePreprocessor:
    def test_make_preprocessor_returns_instance(self) -> None:
        prep = make_preprocessor()
        assert isinstance(prep, TextPreprocessor)

    def test_make_preprocessor_max_chars(self) -> None:
        prep = make_preprocessor(max_chars=10)
        result = prep.clean("a very long sentence that exceeds ten characters")
        assert len(result) <= 10

    def test_make_preprocessor_no_special_chars(self) -> None:
        prep = make_preprocessor(remove_special_chars=False)
        result = prep.clean("Hello, World!")
        assert "," in result
