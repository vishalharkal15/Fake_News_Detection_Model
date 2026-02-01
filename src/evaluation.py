"""
Evaluation Module for Fake News Detection

This module handles:
- Computing classification metrics (accuracy, precision, recall, F1)
- Generating confusion matrices
- Visualizing results
- Detailed classification reports
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    A comprehensive evaluator for the fake news detection model.
    Computes and visualizes various metrics.
    """
    
    def __init__(self, labels: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            labels: List of label names (default: ["FAKE", "REAL"])
        """
        self.labels = labels or ["FAKE", "REAL"]
        self.results = {}
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, pos_label=1)
        metrics["recall"] = recall_score(y_true, y_pred, pos_label=1)
        metrics["f1_score"] = f1_score(y_true, y_pred, pos_label=1)
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        metrics["precision_per_class"] = precision_score(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        metrics["recall_per_class"] = recall_score(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        metrics["f1_per_class"] = f1_score(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        
        # Macro and weighted averages
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        
        # ROC-AUC if probabilities provided
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics["roc_auc"] = auc(fpr, tpr)
            metrics["fpr"] = fpr
            metrics["tpr"] = tpr
        
        self.results = metrics
        return metrics
    
    def print_report(self, metrics: Optional[Dict] = None):
        """
        Print a formatted classification report.
        
        Args:
            metrics: Metrics dictionary (uses self.results if not provided)
        """
        metrics = metrics or self.results
        
        if not metrics:
            logger.error("No metrics to report. Run compute_metrics first.")
            return
        
        print("\n" + "=" * 60)
        print("FAKE NEWS DETECTION - EVALUATION REPORT")
        print("=" * 60)
        
        print("\n📊 OVERALL METRICS:")
        print("-" * 40)
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        if "roc_auc" in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n📈 PER-CLASS METRICS:")
        print("-" * 40)
        print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 40)
        for i, label in enumerate(self.labels):
            print(f"  {label:<10} {metrics['precision_per_class'][i]:.4f}       "
                  f"{metrics['recall_per_class'][i]:.4f}       "
                  f"{metrics['f1_per_class'][i]:.4f}")
        
        print("\n📋 CONFUSION MATRIX:")
        print("-" * 40)
        cm = metrics["confusion_matrix"]
        print(f"  {'':>12} Predicted")
        print(f"  {'':>12} {'FAKE':<8} {'REAL':<8}")
        print(f"  Actual FAKE {cm[0][0]:<8} {cm[0][1]:<8}")
        print(f"  Actual REAL {cm[1][0]:<8} {cm[1][1]:<8}")
        
        # Calculate and display additional insights
        tn, fp, fn, tp = cm.ravel()
        print("\n📍 DETAILED BREAKDOWN:")
        print("-" * 40)
        print(f"  True Negatives (TN):  {tn} (Correctly identified FAKE)")
        print(f"  True Positives (TP):  {tp} (Correctly identified REAL)")
        print(f"  False Positives (FP): {fp} (FAKE classified as REAL)")
        print(f"  False Negatives (FN): {fn} (REAL classified as FAKE)")
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\n  Specificity: {specificity:.4f}")
        print(f"  False Positive Rate: {1 - specificity:.4f}")
        
        print("\n" + "=" * 60)
    
    def plot_confusion_matrix(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot a visual confusion matrix.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save the figure
            figsize: Figure size
        """
        metrics = metrics or self.results
        cm = metrics["confusion_matrix"]
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.labels,
            yticklabels=self.labels,
            annot_kws={"size": 14}
        )
        
        plt.title("Confusion Matrix - Fake News Detection", fontsize=14, pad=20)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot a bar chart comparing metrics.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save the figure
            figsize: Figure size
        """
        metrics = metrics or self.results
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Overall metrics bar chart
        ax1 = axes[0]
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metric_values = [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"]
        ]
        
        bars = ax1.bar(metric_names, metric_values, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
        ax1.set_ylim(0, 1.1)
        ax1.set_title("Overall Metrics", fontsize=12)
        ax1.set_ylabel("Score", fontsize=10)
        
        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                fontsize=10
            )
        
        # Per-class comparison
        ax2 = axes[1]
        x = np.arange(len(self.labels))
        width = 0.25
        
        bars1 = ax2.bar(x - width, metrics["precision_per_class"], width, label="Precision", color="#3498db")
        bars2 = ax2.bar(x, metrics["recall_per_class"], width, label="Recall", color="#9b59b6")
        bars3 = ax2.bar(x + width, metrics["f1_per_class"], width, label="F1 Score", color="#e74c3c")
        
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Per-Class Metrics", fontsize=12)
        ax2.set_ylabel("Score", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.labels)
        ax2.legend(loc="upper right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot ROC curve if probabilities were provided.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save the figure
            figsize: Figure size
        """
        metrics = metrics or self.results
        
        if "fpr" not in metrics or "tpr" not in metrics:
            logger.warning("ROC curve requires prediction probabilities")
            return
        
        plt.figure(figsize=figsize)
        
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            color="#3498db",
            lw=2,
            label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})'
        )
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random Classifier")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to: {save_path}")
        
        plt.show()
    
    def generate_full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Generate a comprehensive evaluation report with all visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary of all metrics
        """
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Print text report
        self.print_report(metrics)
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Generate visualizations
        cm_path = os.path.join(save_dir, "confusion_matrix.png") if save_dir else None
        self.plot_confusion_matrix(metrics, save_path=cm_path)
        
        metrics_path = os.path.join(save_dir, "metrics_comparison.png") if save_dir else None
        self.plot_metrics_comparison(metrics, save_path=metrics_path)
        
        if y_prob is not None:
            roc_path = os.path.join(save_dir, "roc_curve.png") if save_dir else None
            self.plot_roc_curve(metrics, save_path=roc_path)
        
        return metrics


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None
) -> Dict:
    """
    Quick function to evaluate predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.generate_full_report(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob) if y_prob else None
    )


if __name__ == "__main__":
    # Demo with sample data
    print("Evaluation Module Demo")
    print("=" * 40)
    
    # Sample data
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4, 0.95, 0.15, 0.85, 0.25])
    
    evaluator = ModelEvaluator()
    metrics = evaluator.generate_full_report(y_true, y_pred, y_prob)
