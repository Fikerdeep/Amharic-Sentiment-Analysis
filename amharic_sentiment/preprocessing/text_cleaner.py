"""
Text cleaning utilities for Amharic text preprocessing.

This module provides comprehensive text cleaning functions specifically
designed for Amharic language processing, including URL removal,
punctuation handling, and special character cleaning.
"""

import re
from typing import List, Optional


class AmharicTextCleaner:
    """
    A comprehensive text cleaner for Amharic text.

    Handles URL removal, punctuation, special characters, emojis,
    English text, numbers, and various noise in Amharic text data.

    Example:
        >>> cleaner = AmharicTextCleaner()
        >>> clean_text = cleaner.clean("ሰላም!! https://example.com 123")
        >>> print(clean_text)
        'ሰላም'
    """

    # Regex patterns
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    AMHARIC_PUNCTUATION = re.compile(r'[፤።፡፣:,.?/()•""*፨]+')
    SPECIAL_CHARS = re.compile(r"[@#$%^&=?×!,;:_.(){}`'/+*<>\"¤—„®¯™¡\x10»€«·'§"¬¦…""÷~¨©±¥£¶–°•˜'\"|\\\[\]]")
    ENGLISH_AND_NUMBERS = re.compile(r'[a-zA-Z0-9]+')
    ELONGATION = re.compile(r'(.)\1+')
    EMOJI_PATTERN = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    GEEZ_NUMBERS = re.compile(r'[፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻]')
    HTML_TAGS = re.compile(r'<[^>]*>')
    WHITESPACE = re.compile(r'\s+')

    def __init__(
        self,
        remove_urls: bool = True,
        remove_punctuation: bool = True,
        remove_special_chars: bool = True,
        remove_english: bool = True,
        remove_numbers: bool = True,
        remove_emojis: bool = True,
        remove_geez_numbers: bool = True,
        remove_html: bool = True,
        normalize_elongation: bool = True,
        max_elongation: int = 2
    ):
        """
        Initialize the text cleaner with configurable options.

        Args:
            remove_urls: Remove HTTP/HTTPS URLs
            remove_punctuation: Remove Amharic and common punctuation
            remove_special_chars: Remove special characters
            remove_english: Remove English alphabet characters
            remove_numbers: Remove Arabic numerals
            remove_emojis: Remove emoji characters
            remove_geez_numbers: Remove Ge'ez numeral characters
            remove_html: Remove HTML tags
            normalize_elongation: Reduce repeated characters
            max_elongation: Maximum allowed character repetition
        """
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_special_chars = remove_special_chars
        self.remove_english = remove_english
        self.remove_numbers = remove_numbers
        self.remove_emojis = remove_emojis
        self.remove_geez_numbers = remove_geez_numbers
        self.remove_html = remove_html
        self.normalize_elongation = normalize_elongation
        self.max_elongation = max_elongation

    def clean(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        # Remove URLs
        if self.remove_urls:
            cleaned = self.URL_PATTERN.sub('', cleaned)

        # Remove HTML tags
        if self.remove_html:
            cleaned = self.HTML_TAGS.sub('', cleaned)

        # Remove Amharic punctuation
        if self.remove_punctuation:
            cleaned = self.AMHARIC_PUNCTUATION.sub(' ', cleaned)

        # Remove special characters
        if self.remove_special_chars:
            cleaned = self.SPECIAL_CHARS.sub(' ', cleaned)

        # Remove English and numbers
        if self.remove_english or self.remove_numbers:
            cleaned = self.ENGLISH_AND_NUMBERS.sub('', cleaned)

        # Normalize elongation (e.g., "aaaa" -> "aa")
        if self.normalize_elongation:
            replacement = r'\1' * self.max_elongation
            cleaned = self.ELONGATION.sub(replacement, cleaned)

        # Remove emojis
        if self.remove_emojis:
            cleaned = self.EMOJI_PATTERN.sub('', cleaned)

        # Remove Ge'ez numbers
        if self.remove_geez_numbers:
            cleaned = self.GEEZ_NUMBERS.sub('', cleaned)

        # Remove hyphens and dashes
        cleaned = cleaned.replace('-', ' ')

        # Normalize whitespace
        cleaned = self.WHITESPACE.sub(' ', cleaned)

        return cleaned.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of text strings.

        Args:
            texts: List of input texts to clean

        Returns:
            List of cleaned text strings
        """
        return [self.clean(text) for text in texts]

    def __call__(self, text: str) -> str:
        """Allow the cleaner to be called directly."""
        return self.clean(text)


def clean_text(text: str, **kwargs) -> str:
    """
    Convenience function to clean text with default settings.

    Args:
        text: Input text to clean
        **kwargs: Additional arguments passed to AmharicTextCleaner

    Returns:
        Cleaned text string
    """
    cleaner = AmharicTextCleaner(**kwargs)
    return cleaner.clean(text)
