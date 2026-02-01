#!/usr/bin/env python3
"""
Quick Start Script for Fake News Detection

This script provides a simple demonstration of the fake news detection pipeline.
Run this script to quickly test the system with the sample dataset.

Usage:
    python quickstart.py
"""

import os
import sys

# Add the project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def main():
    print("\n" + "=" * 70)
    print("🚀 FAKE NEWS DETECTION - Quick Start Demo")
    print("=" * 70)
    
    # Check if we have a trained model
    model_path = os.path.join(project_dir, "saved_model")
    
    if os.path.exists(model_path):
        print("\n✅ Found trained model. Running inference demo...\n")
        run_inference_demo(model_path)
    else:
        print("\n⚠️  No trained model found.")
        print("   Running training pipeline first...\n")
        run_training_demo()
    
    print("\n" + "=" * 70)
    print("✅ Quick Start Demo Completed!")
    print("=" * 70)

def run_inference_demo(model_path):
    """Run inference demo with existing model."""
    from src.inference import FakeNewsPredictor
    
    # Load model
    print("Loading model...")
    predictor = FakeNewsPredictor(model_path)
    
    # Sample texts
    test_texts = [
        "The Federal Reserve announced a 0.25% interest rate increase today, "
        "citing concerns about inflation in the housing market.",
        
        "SHOCKING: Scientists discover that drinking coffee makes you immortal! "
        "Big Pharma is trying to hide this miracle cure!",
        
        "A new study published in the New England Journal of Medicine found that "
        "moderate exercise can significantly reduce the risk of cardiovascular disease.",
        
        "BREAKING: Famous actor reveals he is actually an alien from another dimension! "
        "Click here before the government removes this post!",
    ]
    
    print("\n📰 Running predictions on sample texts:\n")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        result = predictor.predict(text)
        
        emoji = "✅" if result.prediction == "REAL" else "❌"
        
        print(f"\nExample {i}:")
        print(f"Text: {text[:80]}...")
        print(f"Prediction: {emoji} {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"  - FAKE probability: {result.fake_probability:.2%}")
        print(f"  - REAL probability: {result.real_probability:.2%}")
        print("-" * 60)

def run_training_demo():
    """Run training demo with sample dataset."""
    import subprocess
    
    dataset_path = os.path.join(project_dir, "news_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print("❌ Sample dataset not found!")
        print(f"   Expected at: {dataset_path}")
        return
    
    print("🔧 Starting training with sample dataset...")
    print("   (This is a small demo - for production, use a larger dataset)\n")
    
    # Run training script
    cmd = [
        sys.executable,
        os.path.join(project_dir, "train.py"),
        "--data_path", dataset_path,
        "--output_dir", os.path.join(project_dir, "saved_model"),
        "--epochs", "1",  # Just 1 epoch for demo
        "--batch_size", "8"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
