"""
Data Processing Module for Fake News Detection

This module handles:
- Loading CSV datasets
- Handling missing values
- Encoding labels (REAL=1, FAKE=0)
- Train/test splitting
- Class distribution analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A class to handle all data processing operations for the fake news dataset.
    
    Attributes:
        label_mapping (dict): Maps text labels to numeric values
        reverse_label_mapping (dict): Maps numeric values back to text labels
    """
    
    def __init__(self):
        """Initialize the DataProcessor with label mappings."""
        # REAL = 1, FAKE = 0 as per requirements
        self.label_mapping = {"REAL": 1, "FAKE": 0}
        self.reverse_label_mapping = {1: "REAL", 0: "FAKE"}
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load the dataset from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            pandas DataFrame with the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If required columns are missing
        """
        logger.info(f"Loading dataset from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {len(df)} samples")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        
        # Verify required columns exist
        required_columns = ["label", "text"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - Drop rows with missing text (can't classify empty text)
        - Drop rows with missing labels (need labels for training)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        original_size = len(df)
        
        # Check for missing values
        missing_text = df["text"].isna().sum()
        missing_labels = df["label"].isna().sum()
        
        logger.info(f"Missing text values: {missing_text}")
        logger.info(f"Missing label values: {missing_labels}")
        
        # Drop rows with missing values in critical columns
        df = df.dropna(subset=["text", "label"])
        
        # Also drop empty strings
        df = df[df["text"].str.strip().str.len() > 0]
        
        removed = original_size - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with missing/empty values")
        
        return df.reset_index(drop=True)
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode text labels to numeric values.
        
        REAL -> 1
        FAKE -> 0
        
        Args:
            df: Input DataFrame with 'label' column
            
        Returns:
            DataFrame with encoded labels
        """
        # Standardize labels to uppercase first
        df["label"] = df["label"].str.upper().str.strip()
        
        # Encode using mapping
        df["label_encoded"] = df["label"].map(self.label_mapping)
        
        # Check for any unmapped labels
        unmapped = df["label_encoded"].isna().sum()
        if unmapped > 0:
            unique_labels = df[df["label_encoded"].isna()]["label"].unique()
            logger.warning(f"Found {unmapped} unmapped labels: {unique_labels}")
            # Drop unmapped rows
            df = df.dropna(subset=["label_encoded"])
        
        df["label_encoded"] = df["label_encoded"].astype(int)
        logger.info("Labels encoded successfully (REAL=1, FAKE=0)")
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing (default 0.2 = 20%)
            random_state: Random seed for reproducibility
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            Tuple of (train_df, test_df)
        """
        stratify_col = df["label_encoded"] if stratify else None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def get_class_distribution(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Calculate and display class distribution.
        
        Args:
            df: Input DataFrame
            dataset_name: Name to display in logs
            
        Returns:
            Dictionary with class distribution statistics
        """
        distribution = df["label"].value_counts()
        percentages = df["label"].value_counts(normalize=True) * 100
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{dataset_name} Class Distribution:")
        logger.info(f"{'='*50}")
        
        stats = {}
        for label in distribution.index:
            count = distribution[label]
            pct = percentages[label]
            logger.info(f"  {label}: {count} samples ({pct:.2f}%)")
            stats[label] = {"count": count, "percentage": pct}
        
        # Calculate imbalance ratio
        if len(distribution) == 2:
            imbalance_ratio = distribution.max() / distribution.min()
            logger.info(f"  Imbalance Ratio: {imbalance_ratio:.2f}")
            stats["imbalance_ratio"] = imbalance_ratio
        
        return stats
    
    def process_pipeline(
        self, 
        filepath: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run the complete data processing pipeline.
        
        Args:
            filepath: Path to the CSV file
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df, statistics_dict)
        """
        logger.info("Starting data processing pipeline...")
        
        # Step 1: Load data
        df = self.load_dataset(filepath)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Encode labels
        df = self.encode_labels(df)
        
        # Step 4: Show overall distribution
        overall_stats = self.get_class_distribution(df, "Overall")
        
        # Step 5: Split data
        train_df, test_df = self.split_data(
            df, test_size=test_size, random_state=random_state
        )
        
        # Step 6: Show distribution in splits
        train_stats = self.get_class_distribution(train_df, "Training")
        test_stats = self.get_class_distribution(test_df, "Test")
        
        statistics = {
            "overall": overall_stats,
            "train": train_stats,
            "test": test_stats,
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df)
        }
        
        logger.info("\nData processing pipeline completed successfully!")
        
        return train_df, test_df, statistics


# Utility function for quick processing
def process_news_data(filepath: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick function to process news data.
    
    Args:
        filepath: Path to CSV file
        test_size: Test set proportion
        
    Returns:
        Tuple of (train_df, test_df)
    """
    processor = DataProcessor()
    train_df, test_df, _ = processor.process_pipeline(filepath, test_size)
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module for Fake News Detection")
    print("Import this module and use DataProcessor class or process_news_data function")
