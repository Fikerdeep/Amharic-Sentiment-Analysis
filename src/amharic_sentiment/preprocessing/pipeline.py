"""
Text preprocessing pipeline combining cleaning and normalization.

This module provides a unified pipeline for preprocessing Amharic text,
combining text cleaning and character normalization in a single interface.
"""

from typing import List, Optional, Callable
from amharic_sentiment.preprocessing.text_cleaner import AmharicTextCleaner
from amharic_sentiment.preprocessing.normalizer import AmharicNormalizer


class PreprocessingPipeline:
    """
    A unified preprocessing pipeline for Amharic text.

    Combines text cleaning and character normalization into a single
    configurable pipeline.

    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> processed = pipeline.process("ሰላም!! https://example.com ሃገር")
        >>> print(processed)
        'ሰላም ሀገር'
    """

    def __init__(
        self,
        cleaner: Optional[AmharicTextCleaner] = None,
        normalizer: Optional[AmharicNormalizer] = None,
        custom_processors: Optional[List[Callable[[str], str]]] = None
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            cleaner: Custom text cleaner instance (uses default if None)
            normalizer: Custom normalizer instance (uses default if None)
            custom_processors: List of additional custom processing functions
        """
        self.cleaner = cleaner or AmharicTextCleaner()
        self.normalizer = normalizer or AmharicNormalizer()
        self.custom_processors = custom_processors or []

    def process(self, text: str) -> str:
        """
        Process a single text string through the pipeline.

        Args:
            text: Input text to process

        Returns:
            Processed text string
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Clean text
        processed = self.cleaner.clean(text)

        # Step 2: Normalize characters
        processed = self.normalizer.normalize(processed)

        # Step 3: Apply custom processors
        for processor in self.custom_processors:
            processed = processor(processed)

        return processed

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of text strings.

        Args:
            texts: List of input texts to process

        Returns:
            List of processed text strings
        """
        return [self.process(text) for text in texts]

    def add_processor(self, processor: Callable[[str], str]) -> 'PreprocessingPipeline':
        """
        Add a custom processor to the pipeline.

        Args:
            processor: A callable that takes a string and returns a string

        Returns:
            Self for method chaining
        """
        self.custom_processors.append(processor)
        return self

    def __call__(self, text: str) -> str:
        """Allow the pipeline to be called directly."""
        return self.process(text)


def preprocess_text(text: str, **kwargs) -> str:
    """
    Convenience function to preprocess text with default settings.

    Args:
        text: Input text to process
        **kwargs: Additional arguments for cleaner/normalizer configuration

    Returns:
        Processed text string
    """
    pipeline = PreprocessingPipeline()
    return pipeline.process(text)
