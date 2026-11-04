"""
Model training, evaluation, and persistence module for Phishing Website Detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_NAMES


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    report: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": self.confusion_matrix,
        }


def load_dataset(
    legit_path: str = "legitimate-urls.csv",
    phish_path: str = "phishing-urls.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load legitimate and phishing datasets, clean columns, and split into train/test sets.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if not os.path.exists(legit_path):
        raise FileNotFoundError(f"Legitimate dataset not found at: {legit_path}")
    if not os.path.exists(phish_path):
        raise FileNotFoundError(f"Phishing dataset not found at: {phish_path}")

    legit_df = pd.read_csv(legit_path)
    phish_df = pd.read_csv(phish_path)

    merged = pd.concat([legit_df, phish_df], ignore_index=True)

    # Ensure all required features are present
    missing_features = [col for col in FEATURE_NAMES if col not in merged.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in dataset: {missing_features}")

    X = merged[FEATURE_NAMES]
    y = merged["label"].astype(int)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a Decision Tree Classifier."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest Classifier ensemble."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> EvaluationMetrics:
    """Evaluate model performance and return metric summary."""
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, zero_division=0)

    return EvaluationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm,
        report=report,
    )


def save_model(model: Any, filepath: str) -> None:
    """Save trained model to disk using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Load model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)
