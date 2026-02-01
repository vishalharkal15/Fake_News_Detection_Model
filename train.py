#!/usr/bin/env python3
"""
Fake News Detection - Main Training Script

This is the main entry point for training the fake news detection model.
It orchestrates the entire pipeline:
1. Data loading and processing
2. Text preprocessing
3. Model training with DistilBERT
4. Evaluation
5. Model saving
6. Example predictions

Usage:
    python train.py --data_path news_dataset.csv --output_dir ./saved_model
    
    Or with all options:
    python train.py \
        --data_path news_dataset.csv \
        --output_dir ./saved_model \
        --model_name distilbert-base-uncased \
        --epochs 3 \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --max_length 512
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import our modules
from src.data_processing import DataProcessor
from src.text_preprocessing import TextPreprocessor
from src.model_training import FakeNewsTrainer, ModelConfig
from src.evaluation import ModelEvaluator
from src.inference import FakeNewsPredictor, predict_news, load_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Fake News Detection model using DistilBERT"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="news_dataset.csv",
        help="Path to the CSV dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Early stopping patience (default: 2)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_model",
        help="Directory to save the trained model"
    )
    
    # Other arguments
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run inference (requires existing model)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("\n" + "=" * 70)
    print("🔍 FAKE NEWS DETECTION - Training Pipeline")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70 + "\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # =========================================================================
    # STEP 1: DATA PROCESSING
    # =========================================================================
    logger.info("STEP 1: Loading and processing data...")
    
    data_processor = DataProcessor()
    
    try:
        train_df, test_df, stats = data_processor.process_pipeline(
            filepath=args.data_path,
            test_size=args.test_size,
            random_state=args.seed
        )
    except FileNotFoundError:
        logger.error(f"Dataset not found: {args.data_path}")
        logger.error("Please ensure the CSV file exists with 'text' and 'label' columns.")
        sys.exit(1)
    
    # =========================================================================
    # STEP 2: TEXT PREPROCESSING
    # =========================================================================
    logger.info("\nSTEP 2: Preprocessing text...")
    
    preprocessor = TextPreprocessor()
    
    # Preprocess texts (minimal preprocessing for BERT)
    train_texts = preprocessor.preprocess_batch(train_df["text"].tolist())
    test_texts = preprocessor.preprocess_batch(test_df["text"].tolist())
    
    # Get labels
    train_labels = train_df["label_encoded"].tolist()
    test_labels = test_df["label_encoded"].tolist()
    
    logger.info(f"Preprocessed {len(train_texts)} training samples")
    logger.info(f"Preprocessed {len(test_texts)} test samples")
    
    # =========================================================================
    # STEP 3: MODEL TRAINING
    # =========================================================================
    if not args.skip_training:
        logger.info("\nSTEP 3: Training the model...")
        
        # Create model configuration
        config = ModelConfig(
            model_name=args.model_name,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stopping_patience,
            output_dir="./training_output"
        )
        
        # Print training configuration
        print("\n" + "-" * 50)
        print("Training Configuration:")
        print("-" * 50)
        print(f"  Model: {config.model_name}")
        print(f"  Max Length: {config.max_length}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Early Stopping Patience: {config.early_stopping_patience}")
        print("-" * 50 + "\n")
        
        # Initialize trainer
        trainer = FakeNewsTrainer(config)
        
        # For training, we'll use a validation split from the training data
        # Split train into train/val for early stopping
        from sklearn.model_selection import train_test_split
        
        train_texts_final, val_texts, train_labels_final, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=0.1,  # 10% for validation
            random_state=args.seed,
            stratify=train_labels
        )
        
        logger.info(f"Training samples: {len(train_texts_final)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        
        # Train the model
        train_result = trainer.train(
            train_texts=train_texts_final,
            train_labels=train_labels_final,
            val_texts=val_texts,
            val_labels=val_labels
        )
        
        # =====================================================================
        # STEP 4: SAVE MODEL
        # =====================================================================
        logger.info("\nSTEP 4: Saving the model...")
        
        trainer.save_model(args.output_dir)
        logger.info(f"Model saved to: {args.output_dir}")
    
    else:
        logger.info("\nSkipping training (--skip_training flag set)")
        logger.info(f"Loading existing model from: {args.output_dir}")
    
    # =========================================================================
    # STEP 5: EVALUATION
    # =========================================================================
    logger.info("\nSTEP 5: Evaluating on test set...")
    
    # Load the trained model for prediction
    predictor = FakeNewsPredictor(args.output_dir)
    
    # Make predictions on test set
    logger.info("Making predictions on test set...")
    
    predictions = []
    probabilities = []
    
    # Predict in batches
    batch_size = 32
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        results = predictor.predict_batch(batch_texts)
        
        for result in results:
            predictions.append(1 if result.prediction == "REAL" else 0)
            probabilities.append(result.real_probability)
        
        if (i + batch_size) % 100 == 0:
            logger.info(f"Processed {min(i + batch_size, len(test_texts))}/{len(test_texts)} samples")
    
    # Convert to numpy arrays
    y_true = np.array(test_labels)
    y_pred = np.array(predictions)
    y_prob = np.array(probabilities)
    
    # Evaluate
    evaluator = ModelEvaluator()
    
    # Create output directory for evaluation results
    eval_dir = os.path.join(args.output_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    
    metrics = evaluator.generate_full_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        save_dir=eval_dir
    )
    
    # =========================================================================
    # STEP 6: DEMO PREDICTIONS
    # =========================================================================
    logger.info("\nSTEP 6: Demo predictions...")
    
    print("\n" + "=" * 70)
    print("📰 SAMPLE PREDICTIONS")
    print("=" * 70)
    
    # Demo texts
    demo_texts = [
        "Scientists at NASA have confirmed the discovery of water on Mars, "
        "marking a significant breakthrough in the search for extraterrestrial life.",
        
        "SHOCKING: Local man discovers that eating this one fruit will make "
        "you lose 50 pounds instantly! Doctors hate him!",
        
        "The Federal Reserve announced today that interest rates will remain "
        "unchanged following their quarterly meeting.",
    ]
    
    for i, text in enumerate(demo_texts, 1):
        result = predictor.predict(text)
        
        print(f"\n--- Example {i} ---")
        print(f"Text: {text[:100]}...")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"FAKE probability: {result.fake_probability:.2%}")
        print(f"REAL probability: {result.real_probability:.2%}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED")
    print("=" * 70)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Evaluation results saved to: {eval_dir}")
    print(f"\nFinal Test Set Metrics:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1 Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\n" + "=" * 70)
    
    # Show how to use the model
    print("\n📖 HOW TO USE THE TRAINED MODEL:")
    print("-" * 50)
    print("from src.inference import predict_news, load_predictor")
    print("")
    print("# Load the model")
    print(f"load_predictor('{args.output_dir}')")
    print("")
    print("# Make predictions")
    print("result = predict_news('Your news article text here...')")
    print("print(f\"Prediction: {result['prediction']}\")")
    print("print(f\"Confidence: {result['confidence']:.2%}\")")
    print("-" * 50)


if __name__ == "__main__":
    main()
