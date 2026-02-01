"""
SHAP Explainability Module for Fake News Detection

This module provides:
- SHAP-based model explanations
- Visualization of important words/features
- Understanding which words influence predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeNewsExplainer:
    """
    SHAP-based explainer for the fake news detection model.
    
    Provides interpretability by showing which words/tokens
    most influence the model's predictions.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the explainer.
        
        Args:
            model_path: Path to the saved model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Please install it with: pip install shap"
            )
        
        self.model_path = model_path
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize SHAP explainer
        self._init_explainer()
        
        # Label mappings
        self.id2label = {0: "FAKE", 1: "REAL"}
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _init_explainer(self):
        """Initialize the SHAP explainer."""
        logger.info("Initializing SHAP explainer...")
        
        # Create a prediction function for SHAP
        def predict_proba(texts):
            """Prediction function that returns probabilities."""
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize
            inputs = self.tokenizer(
                list(texts),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            return probs.cpu().numpy()
        
        self.predict_fn = predict_proba
        
        # Create SHAP explainer using the partition explainer for text
        # This is more efficient for transformer models
        self.explainer = shap.Explainer(
            self.predict_fn,
            self.tokenizer,
            output_names=["FAKE", "REAL"]
        )
        
        logger.info("SHAP explainer initialized")
    
    def explain(
        self,
        text: str,
        target_class: Optional[str] = None
    ) -> Dict:
        """
        Generate SHAP explanation for a single text.
        
        Args:
            text: Input text to explain
            target_class: Class to explain ("FAKE" or "REAL", None for predicted)
            
        Returns:
            Dictionary with explanation data
        """
        logger.info("Generating SHAP explanation...")
        
        # Get SHAP values
        shap_values = self.explainer([text])
        
        # Get prediction
        probs = self.predict_fn(text)[0]
        predicted_class_idx = np.argmax(probs)
        predicted_class = self.id2label[predicted_class_idx]
        confidence = probs[predicted_class_idx]
        
        # Determine which class to explain
        if target_class:
            explain_idx = 0 if target_class.upper() == "FAKE" else 1
        else:
            explain_idx = predicted_class_idx
        
        # Extract token-level SHAP values
        # Get the values for the target class
        values = shap_values.values[0, :, explain_idx]
        
        # Get tokens
        tokens = shap_values.data[0]
        
        # Create word importance list
        word_importance = []
        for token, value in zip(tokens, values):
            if token.strip():  # Skip empty tokens
                word_importance.append({
                    "word": token,
                    "importance": float(value),
                    "direction": "supports" if value > 0 else "opposes"
                })
        
        # Sort by absolute importance
        word_importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
        
        return {
            "text": text,
            "prediction": predicted_class,
            "confidence": float(confidence),
            "explained_class": self.id2label[explain_idx],
            "word_importance": word_importance[:20],  # Top 20 words
            "shap_values": shap_values,
            "all_probabilities": {
                "FAKE": float(probs[0]),
                "REAL": float(probs[1])
            }
        }
    
    def visualize(
        self,
        text: str,
        target_class: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Generate and display SHAP visualization.
        
        Args:
            text: Input text to explain
            target_class: Class to explain
            save_path: Path to save the visualization
        """
        explanation = self.explain(text, target_class)
        
        # Print text summary
        print("\n" + "=" * 60)
        print("SHAP EXPLANATION SUMMARY")
        print("=" * 60)
        print(f"\nPrediction: {explanation['prediction']}")
        print(f"Confidence: {explanation['confidence']:.2%}")
        print(f"\nExplaining: Why the model predicts {explanation['explained_class']}")
        
        # Print top important words
        print("\n📊 TOP INFLUENTIAL WORDS:")
        print("-" * 40)
        
        positive_words = []
        negative_words = []
        
        for item in explanation["word_importance"][:15]:
            word = item["word"]
            importance = item["importance"]
            direction = "↑" if importance > 0 else "↓"
            
            if importance > 0:
                positive_words.append((word, importance))
            else:
                negative_words.append((word, importance))
            
            print(f"  {direction} '{word}': {importance:+.4f}")
        
        # Create visualization using SHAP's text plot
        print("\n" + "=" * 60)
        
        # Display SHAP text plot
        shap_values = explanation["shap_values"]
        
        # Use SHAP's built-in text visualization
        shap.plots.text(shap_values[0], display=True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Visualization saved to: {save_path}")
    
    def plot_word_importance(
        self,
        text: str,
        top_n: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Create a bar chart of word importance.
        
        Args:
            text: Input text
            top_n: Number of top words to show
            save_path: Path to save the figure
        """
        explanation = self.explain(text)
        
        # Get top words
        words = []
        importances = []
        colors = []
        
        for item in explanation["word_importance"][:top_n]:
            words.append(item["word"])
            importances.append(item["importance"])
            colors.append("#2ecc71" if item["importance"] > 0 else "#e74c3c")
        
        # Reverse for horizontal bar chart (highest at top)
        words = words[::-1]
        importances = importances[::-1]
        colors = colors[::-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(words, importances, color=colors)
        
        ax.set_xlabel("SHAP Value (Impact on Prediction)", fontsize=12)
        ax.set_title(
            f"Word Importance for {explanation['explained_class']} Prediction\n"
            f"(Actual: {explanation['prediction']}, Confidence: {explanation['confidence']:.1%})",
            fontsize=14
        )
        
        # Add a vertical line at 0
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="Supports prediction"),
            Patch(facecolor="#e74c3c", label="Opposes prediction")
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Word importance plot saved to: {save_path}")
        
        plt.show()
    
    def explain_batch(
        self,
        texts: List[str],
        target_class: Optional[str] = None
    ) -> List[Dict]:
        """
        Explain multiple texts.
        
        Args:
            texts: List of texts to explain
            target_class: Target class to explain
            
        Returns:
            List of explanation dictionaries
        """
        return [self.explain(text, target_class) for text in texts]


def explain_prediction(
    text: str,
    model_path: str,
    visualize: bool = True,
    save_path: Optional[str] = None
) -> Dict:
    """
    Quick function to explain a prediction.
    
    Args:
        text: Text to explain
        model_path: Path to the trained model
        visualize: Whether to show visualizations
        save_path: Path to save visualizations
        
    Returns:
        Explanation dictionary
    """
    explainer = FakeNewsExplainer(model_path)
    explanation = explainer.explain(text)
    
    if visualize:
        explainer.plot_word_importance(text, save_path=save_path)
    
    return explanation


def get_important_words(
    text: str,
    model_path: str,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the most important words for a prediction.
    
    Args:
        text: Input text
        model_path: Path to the model
        top_n: Number of words to return
        
    Returns:
        List of (word, importance) tuples
    """
    explainer = FakeNewsExplainer(model_path)
    explanation = explainer.explain(text)
    
    return [
        (item["word"], item["importance"])
        for item in explanation["word_importance"][:top_n]
    ]


if __name__ == "__main__":
    if SHAP_AVAILABLE:
        print("SHAP Explainability Module")
        print("=" * 50)
        print("\nUsage:")
        print("  from explainability import FakeNewsExplainer, explain_prediction")
        print("")
        print("  # Full explainer")
        print("  explainer = FakeNewsExplainer('./saved_model')")
        print("  explanation = explainer.explain('Your news text...')")
        print("  explainer.plot_word_importance('Your news text...')")
        print("")
        print("  # Quick function")
        print("  explanation = explain_prediction('Your text...', './saved_model')")
    else:
        print("SHAP is not installed. Install with: pip install shap")
