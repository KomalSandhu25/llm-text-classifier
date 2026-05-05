"""Text preprocessing pipeline for the LLM text classification system.

This module provides a configurable ``TextPreprocessor`` that normalises raw
text before tokenisation.  Cleaning steps include URL removal, HTML stripping,
Unicode normalisation, whitespace collapsing, and optional length truncation.
All steps are independently toggleable so the preprocessor can be adapted to
different datasets without subclassing.

Example::

    from src.data.preprocessor import TextPreprocessor

    prep = TextPreprocessor(max_chars=512)
    clean = prep.clean("Visit <b>https://example.com</b> for info!!!")
    # "visit for info"
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    """Hyperparameters that control each cleaning step.

    Attributes:
        lowercase: Convert text to lower case.
        remove_urls: Strip http/https/www URLs.
        remove_html: Unescape HTML entities and strip tags.
        remove_special_chars: Replace non-alphanumeric, non-space characters
            with a single space.
        normalize_unicode: Apply NFKC normalisation to collapse compatibility
            characters (e.g. full-width digits → ASCII digits).
        collapse_whitespace: Replace runs of whitespace (including newlines and
            tabs) with a single space and strip leading / trailing whitespace.
        max_chars: Hard character-limit applied *after* all other steps.  Set
            to ``None`` to disable truncation.
    """

    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_special_chars: bool = True
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    max_chars: Optional[int] = None


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """Stateless text cleaning pipeline.

    Each ``clean`` call applies a deterministic sequence of transformations
    controlled by ``PreprocessorConfig``.  The preprocessor is safe to share
    across threads because it holds no mutable state after construction.

    Args:
        config: Preprocessing configuration.  Keyword arguments are forwarded
            to ``PreprocessorConfig`` when *config* is omitted.
        max_chars: Convenience shortcut — sets ``config.max_chars`` when
            *config* is not provided.

    Example::

        prep = TextPreprocessor(max_chars=256)
        prep.clean("Hello  WORLD! <br>  https://t.co/abc")
        # "hello world"
    """

    def __init__(
        self,
        config: Optional[PreprocessorConfig] = None,
        *,
        max_chars: Optional[int] = None,
    ) -> None:
        if config is None:
            config = PreprocessorConfig(max_chars=max_chars)
        self.config: PreprocessorConfig = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Apply all enabled cleaning steps to *text*.

        Args:
            text: Raw input string (may be empty).

        Returns:
            Cleaned string.  Never ``None``; returns ``""`` for falsy input.

        Example::

            prep = TextPreprocessor()
            prep.clean("  Visit <b>https://foo.com</b> NOW!!! ")
            # "visit now"
        """
        if not text:
            return ""

        cfg = self.config

        if cfg.normalize_unicode:
            text = self._normalize_unicode(text)

        if cfg.remove_html:
            text = self._remove_html(text)

        if cfg.remove_urls:
            text = self._remove_urls(text)

        if cfg.remove_special_chars:
            text = self._remove_special_chars(text)

        if cfg.lowercase:
            text = text.lower()

        if cfg.collapse_whitespace:
            text = self._collapse_whitespace(text)

        if cfg.max_chars is not None and len(text) > cfg.max_chars:
            text = self._truncate(text, cfg.max_chars)

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Apply :meth:`clean` to every element of *texts*.

        Args:
            texts: List of raw strings.

        Returns:
            List of cleaned strings with the same length as *texts*.
        """
        return [self.clean(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Apply NFKC Unicode normalisation.

        Args:
            text: Input string.

        Returns:
            Normalised string.
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        """Unescape HTML entities, then remove all HTML tags.

        Args:
            text: Input string potentially containing HTML markup.

        Returns:
            Plain text with entities decoded and tags stripped.
        """
        text = html.unescape(text)          # &amp; → &, &lt; → <, …
        text = _HTML_TAG_RE.sub(" ", text)  # <br/>, <b>, … → space
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Replace URL tokens with a single space.

        Args:
            text: Input string possibly containing http/https/www URLs.

        Returns:
            String with URLs removed.
        """
        return _URL_RE.sub(" ", text)

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Replace non-alphanumeric, non-space characters with a space.

        Args:
            text: Input string.

        Returns:
            String retaining only letters, digits, and whitespace.
        """
        return _SPECIAL_CHAR_RE.sub(" ", text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse consecutive whitespace to a single space and strip.

        Args:
            text: Input string.

        Returns:
            Stripped string with internal whitespace collapsed.
        """
        return _WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate *text* to *max_chars* characters at the nearest word boundary.

        Prefers breaking at a whitespace so that the last token is not cut in
        half.  Falls back to a hard character cut when no space is found within
        the limit.

        Args:
            text: Input string longer than *max_chars*.
            max_chars: Maximum number of characters to retain.

        Returns:
            Truncated string, stripped of trailing whitespace.
        """
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated.rstrip()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_preprocessor(
    max_chars: Optional[int] = None,
    remove_special_chars: bool = True,
) -> TextPreprocessor:
    """Convenience factory that returns a ``TextPreprocessor`` with sensible defaults.

    Args:
        max_chars: Optional character truncation limit.
        remove_special_chars: Whether to strip non-alphanumeric characters.

    Returns:
        Configured ``TextPreprocessor`` instance.
    """
    config = PreprocessorConfig(
        max_chars=max_chars,
        remove_special_chars=remove_special_chars,
    )
    return TextPreprocessor(config=config)
