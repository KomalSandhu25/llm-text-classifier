"""Text data augmentation strategies for the classification pipeline.

Two augmentation techniques are provided:

1. **Synonym Replacement** (``SynonymAugmenter``) — replaces a random subset
   of content words with WordNet synonyms.  Fully implemented with NLTK.

2. **Back-Translation** (``BackTranslationAugmenter``) — structural placeholder
   that defines the interface and raises ``NotImplementedError`` by default.
   Callers can subclass it and inject an API client (e.g. DeepL, Google
   Translate) to enable it in production.

Both augmenters share the ``TextAugmenter`` abstract base, allowing them to be
used interchangeably in a pipeline.

Example::

    from src.data.augmentation import SynonymAugmenter

    aug = SynonymAugmenter(replace_prob=0.2, seed=42)
    original = "The government announced a new economic policy"
    augmented = aug.augment(original)
    # "The authorities announced a new economic policy"  (may vary)
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK lazy initialisation
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if they are not already present.

    Downloads are cached locally after the first call, so subsequent calls
    are effectively free.
    """
    import nltk  # noqa: PLC0415 – deferred import to keep startup fast

    resources = [
        ("corpora", "wordnet"),
        ("corpora", "omw-1.4"),
        ("taggers", "averaged_perceptron_tagger"),
        ("tokenizers", "punkt"),
    ]
    for resource_type, resource_name in resources:
        try:
            nltk.data.find(f"{resource_type}/{resource_name}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource_name)
            nltk.download(resource_name, quiet=True)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TextAugmenter(ABC):
    """Abstract base class for text augmentation strategies.

    All concrete augmenters must implement :meth:`augment`.  The :meth:`augment_batch`
    method is provided for convenience and delegates to :meth:`augment`.

    Args:
        seed: Optional integer seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def augment(self, text: str) -> str:
        """Return an augmented version of *text*.

        Args:
            text: Input string.

        Returns:
            Augmented string.  May equal *text* when no substitution is made.
        """
        ...

    def augment_batch(self, texts: list[str]) -> list[str]:
        """Apply :meth:`augment` to each element of *texts*.

        Args:
            texts: List of input strings.

        Returns:
            List of augmented strings with the same length as *texts*.
        """
        return [self.augment(t) for t in texts]


# ---------------------------------------------------------------------------
# Synonym replacement
# ---------------------------------------------------------------------------

class SynonymAugmenter(TextAugmenter):
    """Replace content words with WordNet synonyms at random.

    Only nouns and verbs are eligible for replacement; stopwords, punctuation,
    and named entities are left unchanged to preserve sentence meaning.

    The replacement probability is applied independently per *eligible* token,
    so the expected number of replacements scales naturally with sentence length.

    Args:
        replace_prob: Probability of replacing each eligible word.
        max_synonyms_to_consider: Maximum number of synonyms sampled from
            WordNet per word — using the highest-frequency synonym.
        seed: Random seed for reproducibility.

    Example::

        aug = SynonymAugmenter(replace_prob=0.3, seed=0)
        aug.augment("Scientists discovered a new planet beyond Neptune")
        # may return "Scientists discovered a new satellite beyond Neptune"
    """

    # POS tags treated as eligible for synonym replacement
    _ELIGIBLE_POS: frozenset[str] = frozenset({"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})

    def __init__(
        self,
        replace_prob: float = 0.15,
        max_synonyms_to_consider: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= replace_prob <= 1.0:
            raise ValueError(f"replace_prob must be in [0, 1], got {replace_prob}")
        self.replace_prob = replace_prob
        self.max_synonyms_to_consider = max_synonyms_to_consider
        _ensure_nltk_resources()

    def augment(self, text: str) -> str:
        """Replace eligible words with WordNet synonyms.

        Args:
            text: Input sentence.

        Returns:
            Augmented sentence.  Returns *text* unchanged when NLTK is
            unavailable or when no synonyms are found.
        """
        if not text.strip():
            return text

        import nltk  # noqa: PLC0415
        from nltk.corpus import wordnet  # noqa: PLC0415

        tokens: list[str] = nltk.word_tokenize(text)
        pos_tags: list[tuple[str, str]] = nltk.pos_tag(tokens)

        augmented: list[str] = []
        for word, pos in pos_tags:
            if pos in self._ELIGIBLE_POS and self._rng.random() < self.replace_prob:
                synonym = self._get_synonym(word, wordnet)
                augmented.append(synonym if synonym else word)
            else:
                augmented.append(word)

        return self._detokenize(augmented)

    def _get_synonym(self, word: str, wordnet) -> Optional[str]:
        """Return a random WordNet synonym for *word*, or ``None``.

        Args:
            word: Source word (lower-cased internally before lookup).
            wordnet: ``nltk.corpus.wordnet`` module reference.

        Returns:
            A synonym string that differs from *word*, or ``None`` if no
            synonym is available.
        """
        synsets = wordnet.synsets(word.lower())
        if not synsets:
            return None

        synonyms: list[str] = []
        for synset in synsets[: self.max_synonyms_to_consider]:
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower():
                    synonyms.append(candidate)

        if not synonyms:
            return None

        return self._rng.choice(synonyms)

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        """Naïve detokenisation — joins tokens with spaces, contracts punctuation.

        Args:
            tokens: List of string tokens produced by ``nltk.word_tokenize``.

        Returns:
            Reconstructed sentence.
        """
        text = " ".join(tokens)
        # Contract spaces before punctuation
        import re
        text = re.sub(r" ([.,!?;:'\")])", r"\1", text)
        text = re.sub(r"([(\"]) ", r"\1", text)
        return text


# ---------------------------------------------------------------------------
# Back-translation placeholder
# ---------------------------------------------------------------------------

class BackTranslationAugmenter(TextAugmenter):
    """Back-translation augmentation — interface definition.

    Back-translation typically routes text through an intermediate language
    (e.g. English → German → English) using a machine-translation service.
    This class defines the expected interface so that experiments can switch
    to a live implementation without changing downstream code.

    **Production usage**: subclass ``BackTranslationAugmenter`` and override
    :meth:`_translate` with calls to your preferred MT API
    (DeepL, Google Translate, Helsinki-NLP OPUS-MT, etc.).

    Args:
        intermediate_lang: BCP-47 language code of the pivot language.
        seed: Random seed (unused in the base implementation).

    Example::

        class DeepLBackTranslation(BackTranslationAugmenter):
            def __init__(self, api_key: str, **kwargs):
                super().__init__(**kwargs)
                import deepl
                self._client = deepl.Translator(api_key)

            def _translate(self, text: str, target_lang: str) -> str:
                return self._client.translate_text(text, target_lang=target_lang).text

        aug = DeepLBackTranslation(api_key="...", intermediate_lang="DE")
        aug.augment("The stock market rallied on positive earnings reports")
    """

    def __init__(
        self,
        intermediate_lang: str = "de",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.intermediate_lang = intermediate_lang

    def augment(self, text: str) -> str:
        """Augment *text* via back-translation.

        Args:
            text: English input sentence.

        Returns:
            Back-translated English string.

        Raises:
            NotImplementedError: Always, in the base implementation.  Subclass
                and override :meth:`_translate` to enable.
        """
        pivot = self._translate(text, target_lang=self.intermediate_lang)
        return self._translate(pivot, target_lang="en")

    def _translate(self, text: str, target_lang: str) -> str:
        """Call an external translation service.

        Args:
            text: Input text in the source language.
            target_lang: BCP-47 target language code (e.g. ``"de"``, ``"en"``).

        Returns:
            Translated string.

        Raises:
            NotImplementedError: Must be implemented by a subclass that wires
                up a real translation API.
        """
        raise NotImplementedError(
            "BackTranslationAugmenter._translate() must be overridden. "
            "Subclass this class and implement _translate() with your preferred "
            "MT API (DeepL, Google Translate, Helsinki-NLP OPUS-MT, …)."
        )


# ---------------------------------------------------------------------------
# Composite pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Apply multiple augmenters in sequence with a global enable/disable switch.

    Each augmenter is applied independently; the output of the *i*-th augmenter
    becomes the input to the *(i+1)*-th.  Use ``apply_prob`` to randomly skip
    the entire pipeline for a given sample, which retains the original text a
    fraction ``(1 - apply_prob)`` of the time.

    Args:
        augmenters: Ordered list of ``TextAugmenter`` instances.
        apply_prob: Probability of applying the pipeline to any given sample.
        seed: Random seed.

    Example::

        pipeline = AugmentationPipeline(
            augmenters=[SynonymAugmenter(replace_prob=0.2, seed=0)],
            apply_prob=0.5,
        )
        result = pipeline.augment("Scientists unveiled a groundbreaking climate model")
    """

    def __init__(
        self,
        augmenters: list[TextAugmenter],
        apply_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self._augmenters = augmenters
        self.apply_prob = apply_prob
        self._rng = random.Random(seed)

    def augment(self, text: str) -> str:
        """Run all augmenters on *text* with probability ``apply_prob``.

        Args:
            text: Input string.

        Returns:
            Augmented (or original) string.
        """
        if self._rng.random() > self.apply_prob:
            return text
        for aug in self._augmenters:
            text = aug.augment(text)
        return text

    def augment_batch(self, texts: list[str]) -> list[str]:
        """Apply :meth:`augment` to each element of *texts*.

        Args:
            texts: List of input strings.

        Returns:
            List of augmented strings.
        """
        return [self.augment(t) for t in texts]
