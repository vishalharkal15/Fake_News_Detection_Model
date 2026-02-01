"""
Inference Module for Fake News Detection

This module provides:
- predict_news() function for single predictions
- Batch prediction capabilities
- Confidence scores
- Easy-to-use interface for production deployment
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import our preprocessing module
try:
    from .text_preprocessing import TextPreprocessor
except ImportError:
    from text_preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    text: str
    prediction: str  # "FAKE" or "REAL"
    confidence: float  # Confidence score (0-1)
    fake_probability: float  # Probability of being FAKE
    real_probability: float  # Probability of being REAL
    
    def __repr__(self):
        return (f"PredictionResult(prediction='{self.prediction}', "
                f"confidence={self.confidence:.4f})")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "fake_probability": round(self.fake_probability, 4),
            "real_probability": round(self.real_probability, 4)
        }


class FakeNewsPredictor:
    """
    Production-ready predictor for fake news detection.
    
    Usage:
        predictor = FakeNewsPredictor("./saved_model")
        result = predictor.predict("Some news article text...")
        print(result.prediction, result.confidence)
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Load model and tokenizer
        self._load_model()
        
        # Label mappings
        self.id2label = {0: "FAKE", 1: "REAL"}
        self.label2id = {"FAKE": 0, "REAL": 1}
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before prediction."""
        return self.preprocessor(text)
    
    def _tokenize(self, text: str) -> Dict:
        """Tokenize text for the model."""
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
    
    @torch.no_grad()
    def predict(self, text: str) -> PredictionResult:
        """
        Predict whether a news article is fake or real.
        
        Args:
            text: The news article text to classify
            
        Returns:
            PredictionResult with prediction and confidence score
        """
        # Preprocess
        processed_text = self._preprocess(text)
        
        # Tokenize
        inputs = self._tokenize(processed_text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model output
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)[0]
        fake_prob = probabilities[0].item()
        real_prob = probabilities[1].item()
        
        # Get prediction
        predicted_class = torch.argmax(probabilities).item()
        prediction = self.id2label[predicted_class]
        confidence = probabilities[predicted_class].item()
        
        return PredictionResult(
            text=text,
            prediction=prediction,
            confidence=confidence,
            fake_probability=fake_prob,
            real_probability=real_prob
        )
    
    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """
        Predict for multiple texts at once.
        
        Args:
            texts: List of news article texts
            
        Returns:
            List of PredictionResult objects
        """
        # Preprocess all texts
        processed_texts = [self._preprocess(t) for t in texts]
        
        # Tokenize as batch
        inputs = self.tokenizer(
            processed_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Create results
        results = []
        for i, text in enumerate(texts):
            probs = probabilities[i]
            fake_prob = probs[0].item()
            real_prob = probs[1].item()
            
            predicted_class = torch.argmax(probs).item()
            prediction = self.id2label[predicted_class]
            confidence = probs[predicted_class].item()
            
            results.append(PredictionResult(
                text=text,
                prediction=prediction,
                confidence=confidence,
                fake_probability=fake_prob,
                real_probability=real_prob
            ))
        
        return results
    
    def get_probabilities(self, text: str) -> Tuple[float, float]:
        """
        Get raw probabilities for a text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (fake_probability, real_probability)
        """
        result = self.predict(text)
        return result.fake_probability, result.real_probability


# Global predictor instance for easy access
_predictor: Optional[FakeNewsPredictor] = None


def load_predictor(model_path: str) -> FakeNewsPredictor:
    """
    Load a predictor and store it globally.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        FakeNewsPredictor instance
    """
    global _predictor
    _predictor = FakeNewsPredictor(model_path)
    return _predictor


def predict_news(text: str, model_path: Optional[str] = None) -> Dict:
    """
    Main function to predict if news is fake or real.
    
    This is the primary interface for making predictions.
    
    Args:
        text: The news article text to classify
        model_path: Path to the model (only needed on first call)
        
    Returns:
        Dictionary with:
        - prediction: "FAKE" or "REAL"
        - confidence: Confidence score (0-1)
        - fake_probability: Probability of being fake
        - real_probability: Probability of being real
        
    Example:
        >>> result = predict_news("Breaking news: Scientists discover...")
        >>> print(f"Prediction: {result['prediction']}")
        >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    global _predictor
    
    # Load model if not already loaded
    if _predictor is None:
        if model_path is None:
            raise ValueError(
                "Model not loaded. Please provide model_path on first call, "
                "or use load_predictor() first."
            )
        load_predictor(model_path)
    
    # Make prediction
    result = _predictor.predict(text)
    
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
        "fake_probability": result.fake_probability,
        "real_probability": result.real_probability
    }


def predict_news_batch(texts: List[str], model_path: Optional[str] = None) -> List[Dict]:
    """
    Predict for multiple texts at once.
    
    Args:
        texts: List of news articles
        model_path: Path to the model (only needed on first call)
        
    Returns:
        List of prediction dictionaries
    """
    global _predictor
    
    if _predictor is None:
        if model_path is None:
            raise ValueError("Model not loaded. Please provide model_path.")
        load_predictor(model_path)
    
    results = _predictor.predict_batch(texts)
    return [r.to_dict() for r in results]


if __name__ == "__main__":
    print("Inference Module for Fake News Detection")
    print("=" * 50)
    print("\nUsage:")
    print("  from inference import predict_news, load_predictor")
    print("  ")
    print("  # Option 1: Load once, predict many")
    print("  load_predictor('./saved_model')")
    print("  result = predict_news('Your news text here...')")
    print("  ")
    print("  # Option 2: Auto-load on first prediction")
    print("  result = predict_news('Your news text...', model_path='./saved_model')")
    print("  ")
    print("  print(f\"Prediction: {result['prediction']}\")")
    print("  print(f\"Confidence: {result['confidence']:.2%}\")")
