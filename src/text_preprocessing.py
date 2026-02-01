"""
Text Preprocessing Module for Fake News Detection

This module handles minimal text preprocessing optimized for transformer models:
- URL removal
- Space normalization
- Preserves punctuation (important for BERT/DistilBERT)
- Preserves casing (important for uncased models to handle properly)
"""

import re
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    A class for preprocessing text for transformer-based models.
    
    Note: We use MINIMAL preprocessing because:
    1. BERT/DistilBERT are trained on natural text with punctuation
    2. Punctuation carries semantic meaning (e.g., "!" vs ".")
    3. The tokenizer handles special characters appropriately
    """
    
    def __init__(self, remove_urls: bool = True, normalize_spaces: bool = True):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            remove_urls: Whether to remove URLs from text
            normalize_spaces: Whether to normalize whitespace
        """
        self.remove_urls = remove_urls
        self.normalize_spaces = normalize_spaces
        
        # Compile regex patterns for efficiency
        # Pattern to match various URL formats
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        )
        
        # Pattern for multiple spaces
        self.multi_space_pattern = re.compile(r'\s+')
    
    def remove_urls_from_text(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        return self.url_pattern.sub(' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        - Replaces multiple spaces with single space
        - Strips leading/trailing whitespace
        - Handles tabs, newlines, etc.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace all whitespace sequences with single space
        text = self.multi_space_pattern.sub(' ', text)
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Minimal preprocessing for transformer models:
        1. Remove URLs (they don't carry useful semantic info)
        2. Normalize spaces
        3. Keep punctuation and casing intact!
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Step 1: Remove URLs if enabled
        if self.remove_urls:
            text = self.remove_urls_from_text(text)
        
        # Step 2: Normalize whitespace if enabled
        if self.normalize_spaces:
            text = self.normalize_whitespace(text)
        
        # NOTE: We intentionally do NOT:
        # - Remove punctuation (BERT uses it for context)
        # - Convert to lowercase (let the tokenizer handle this)
        # - Remove special characters (may carry meaning)
        # - Stem or lemmatize (BERT uses subword tokenization)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Make the preprocessor callable for convenience.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Preprocessed text(s)
        """
        if isinstance(text, list):
            return self.preprocess_batch(text)
        return self.preprocess(text)


# Convenience function for quick preprocessing
def clean_text(text: str) -> str:
    """
    Quick function to clean a single text for transformer models.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)


def clean_texts(texts: List[str]) -> List[str]:
    """
    Quick function to clean multiple texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of cleaned texts
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_batch(texts)


if __name__ == "__main__":
    # Demo
    sample_texts = [
        "BREAKING: New study shows https://example.com/study interesting results!",
        "Visit www.fakenews.com for more   information    about this story.",
        "Scientists say THIS is AMAZING! Check it out at http://bit.ly/abc123",
    ]
    
    preprocessor = TextPreprocessor()
    
    print("Text Preprocessing Demo")
    print("=" * 60)
    
    for original in sample_texts:
        cleaned = preprocessor(original)
        print(f"\nOriginal: {original}")
        print(f"Cleaned:  {cleaned}")
