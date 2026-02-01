"""
Model Training Module for Fake News Detection

This module handles:
- DistilBERT/BERT model setup
- Tokenization
- Training with HuggingFace Trainer API
- Early stopping
- Model saving
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the model training."""
    model_name: str = "distilbert-base-uncased"  # Can also use "bert-base-uncased"
    max_length: int = 512  # Maximum sequence length
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    output_dir: str = "./model_output"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500


class FakeNewsDataset:
    """
    Custom dataset class for fake news detection.
    Handles tokenization and creates HuggingFace Dataset objects.
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Initialize the dataset handler.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: Dictionary with 'text' key containing texts to tokenize
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False  # We'll use dynamic padding with DataCollator
        )
    
    def create_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """
        Create a HuggingFace Dataset from texts and labels.
        
        Args:
            texts: List of text strings
            labels: List of integer labels (0 or 1)
            
        Returns:
            HuggingFace Dataset object
        """
        # Create dataset from lists
        dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })
        
        # Tokenize the dataset
        dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"],  # Remove text column after tokenization
            desc="Tokenizing"
        )
        
        return dataset


class FakeNewsTrainer:
    """
    Main trainer class for the fake news detection model.
    Uses HuggingFace Transformers with DistilBERT/BERT.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            config: ModelConfig object with training parameters
        """
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Label mappings
        self.id2label = {0: "FAKE", 1: "REAL"}
        self.label2id = {"FAKE": 0, "REAL": 1}
    
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info("Model and tokenizer loaded successfully")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
    
    def compute_metrics(self, eval_pred) -> Dict:
        """
        Compute metrics for evaluation.
        
        Args:
            eval_pred: EvalPrediction object with predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def prepare_datasets(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int]
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            train_texts: Training text samples
            train_labels: Training labels
            val_texts: Validation text samples
            val_labels: Validation labels
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        dataset_handler = FakeNewsDataset(self.tokenizer, self.config.max_length)
        
        logger.info("Preparing training dataset...")
        train_dataset = dataset_handler.create_dataset(train_texts, train_labels)
        
        logger.info("Preparing validation dataset...")
        val_dataset = dataset_handler.create_dataset(val_texts, val_labels)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_texts: Training text samples
            train_labels: Training labels
            val_texts: Validation text samples
            val_labels: Validation labels
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(
            train_texts, train_labels, val_texts, val_labels
        )
        
        # Create data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Set up early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=0.001
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Train the model
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Early stopping patience: {self.config.early_stopping_patience}")
        
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Log training results
        logger.info("\nTraining completed!")
        logger.info(f"  Training loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_texts: Test text samples
            test_labels: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.trainer is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        dataset_handler = FakeNewsDataset(self.tokenizer, self.config.max_length)
        test_dataset = dataset_handler.create_dataset(test_texts, test_labels)
        
        logger.info("Evaluating model on test set...")
        results = self.trainer.evaluate(test_dataset)
        
        return results
    
    def save_model(self, save_path: str):
        """
        Save the trained model and tokenizer.
        
        Args:
            save_path: Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer saved to: {save_path}")
        
        # Save config
        config_path = os.path.join(save_path, "training_config.txt")
        with open(config_path, "w") as f:
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Max length: {self.config.max_length}\n")
            f.write(f"Learning rate: {self.config.learning_rate}\n")
            f.write(f"Epochs: {self.config.num_epochs}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
        
        logger.info(f"Training config saved to: {config_path}")
    
    @classmethod
    def load_trained_model(cls, model_path: str) -> 'FakeNewsTrainer':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            FakeNewsTrainer instance with loaded model
        """
        trainer = cls()
        
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_path)
        trainer.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        trainer.model.to(trainer.device)
        
        logger.info(f"Model loaded from: {model_path}")
        
        return trainer


if __name__ == "__main__":
    print("Model Training Module for Fake News Detection")
    print("Import this module and use FakeNewsTrainer class")
