"""
Amharic character normalization utilities.

This module handles the normalization of Amharic character variants,
including different Unicode representations of the same character
and labialized character combinations.
"""

import re
from typing import List, Dict, Tuple


class AmharicNormalizer:
    """
    Normalizer for Amharic character-level variations.

    Handles:
    - Character variant normalization (e.g., different forms of the same letter)
    - Labialized character combinations
    - Unicode normalization for Ge'ez script

    Example:
        >>> normalizer = AmharicNormalizer()
        >>> normalized = normalizer.normalize("ሃገር")  # Different form of ሀ
        >>> print(normalized)
        'ሀገር'
    """

    # Character variant mappings
    # Maps variant characters to their normalized form
    CHAR_VARIANTS: List[Tuple[str, str]] = [
        # ሀ family
        ('[ሃኅኃሐሓኻ]', 'ሀ'),
        ('[ሑኁዅ]', 'ሁ'),
        ('[ኂሒኺ]', 'ሂ'),
        ('[ኌሔዄ]', 'ሄ'),
        ('[ሕኅ]', 'ህ'),
        ('[ኆሖኾ]', 'ሆ'),

        # ሰ family (ሠ variants)
        ('[ሠ]', 'ሰ'),
        ('[ሡ]', 'ሱ'),
        ('[ሢ]', 'ሲ'),
        ('[ሣ]', 'ሳ'),
        ('[ሤ]', 'ሴ'),
        ('[ሥ]', 'ስ'),
        ('[ሦ]', 'ሶ'),

        # አ family (ዐ variants)
        ('[ዓኣዐ]', 'አ'),
        ('[ዑ]', 'ኡ'),
        ('[ዒ]', 'ኢ'),
        ('[ዔ]', 'ኤ'),
        ('[ዕ]', 'እ'),
        ('[ዖ]', 'ኦ'),

        # ፀ family (ጸ variants)
        ('[ጸ]', 'ፀ'),
        ('[ጹ]', 'ፁ'),
        ('[ጺ]', 'ፂ'),
        ('[ጻ]', 'ፃ'),
        ('[ጼ]', 'ፄ'),
        ('[ጽ]', 'ፅ'),
        ('[ጾ]', 'ፆ'),

        # Additional variants
        ('[ቊ]', 'ቁ'),
        ('[ኵ]', 'ኩ'),
    ]

    # Labialized character combinations
    # Maps two-character sequences to their single labialized form
    LABIALIZED_MAPPINGS: List[Tuple[str, str]] = [
        ('(ሉ[ዋአ])', 'ሏ'),
        ('(ሙ[ዋአ])', 'ሟ'),
        ('(ቱ[ዋአ])', 'ቷ'),
        ('(ሩ[ዋአ])', 'ሯ'),
        ('(ሱ[ዋአ])', 'ሷ'),
        ('(ሹ[ዋአ])', 'ሿ'),
        ('(ቁ[ዋአ])', 'ቋ'),
        ('(ቡ[ዋአ])', 'ቧ'),
        ('(ቹ[ዋአ])', 'ቿ'),
        ('(ሁ[ዋአ])', 'ኋ'),
        ('(ኑ[ዋአ])', 'ኗ'),
        ('(ኙ[ዋአ])', 'ኟ'),
        ('(ኩ[ዋአ])', 'ኳ'),
        ('(ዙ[ዋአ])', 'ዟ'),
        ('(ጉ[ዋአ])', 'ጓ'),
        ('(ደ[ዋአ])', 'ዷ'),
        ('(ጡ[ዋአ])', 'ጧ'),
        ('(ጩ[ዋአ])', 'ጯ'),
        ('(ጹ[ዋአ])', 'ጿ'),
        ('(ፉ[ዋአ])', 'ፏ'),
    ]

    def __init__(
        self,
        normalize_variants: bool = True,
        normalize_labialized: bool = True
    ):
        """
        Initialize the normalizer.

        Args:
            normalize_variants: Normalize character variants
            normalize_labialized: Normalize labialized character combinations
        """
        self.normalize_variants = normalize_variants
        self.normalize_labialized = normalize_labialized

        # Compile regex patterns for efficiency
        self._variant_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.CHAR_VARIANTS
        ]
        self._labialized_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.LABIALIZED_MAPPINGS
        ]

    def normalize(self, text: str) -> str:
        """
        Normalize Amharic text.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text string
        """
        if not text or not isinstance(text, str):
            return ""

        normalized = text

        # Apply character variant normalization
        if self.normalize_variants:
            for pattern, replacement in self._variant_patterns:
                normalized = pattern.sub(replacement, normalized)

        # Apply labialized character normalization
        if self.normalize_labialized:
            for pattern, replacement in self._labialized_patterns:
                normalized = pattern.sub(replacement, normalized)

        return normalized

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normalize a batch of text strings.

        Args:
            texts: List of input texts to normalize

        Returns:
            List of normalized text strings
        """
        return [self.normalize(text) for text in texts]

    def __call__(self, text: str) -> str:
        """Allow the normalizer to be called directly."""
        return self.normalize(text)


def normalize_amharic(text: str, **kwargs) -> str:
    """
    Convenience function to normalize Amharic text with default settings.

    Args:
        text: Input text to normalize
        **kwargs: Additional arguments passed to AmharicNormalizer

    Returns:
        Normalized text string
    """
    normalizer = AmharicNormalizer(**kwargs)
    return normalizer.normalize(text)
