"""
Phishing Website Detection package.
"""

from .features import FEATURE_NAMES, FeatureExtraction
from .model import (
    EvaluationMetrics,
    evaluate_model,
    load_dataset,
    load_model,
    save_model,
    train_decision_tree,
    train_random_forest,
)
from .predictor import PhishingPredictor, PredictionResult

__all__ = [
    "FeatureExtraction",
    "FEATURE_NAMES",
    "EvaluationMetrics",
    "load_dataset",
    "train_decision_tree",
    "train_random_forest",
    "evaluate_model",
    "save_model",
    "load_model",
    "PhishingPredictor",
    "PredictionResult",
]
