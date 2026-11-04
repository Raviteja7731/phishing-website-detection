#!/usr/bin/env python3
"""
CLI Training Script for Phishing Website Detection.

Loads the datasets, trains Decision Tree and Random Forest models,
evaluates performance metrics, and exports modern serialized model files.
"""

import argparse
import pickle
import sys
from src.model import (
    evaluate_model,
    load_dataset,
    save_model,
    train_decision_tree,
    train_random_forest,
)


def main():
    parser = argparse.ArgumentParser(description="Train Phishing Website Detection models.")
    parser.add_argument(
        "--legit",
        default="legitimate-urls.csv",
        help="Path to legitimate URLs CSV dataset",
    )
    parser.add_argument(
        "--phish",
        default="phishing-urls.csv",
        help="Path to phishing URLs CSV dataset",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split fraction (default: 0.20)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PHISHING WEBSITE DETECTION: MODEL TRAINING PIPELINE")
    print("=" * 60)

    print(f"Loading datasets from:\n - {args.legit}\n - {args.phish}")
    X_train, X_test, y_train, y_test = load_dataset(
        legit_path=args.legit,
        phish_path=args.phish,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")
    print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}\n")

    # 1. Decision Tree Model
    print("--- 1. Training Decision Tree Classifier ---")
    dt_model = train_decision_tree(X_train, y_train, random_state=args.random_state)
    dt_metrics = evaluate_model(dt_model, X_test, y_test)
    print(f"Decision Tree Accuracy : {dt_metrics.accuracy * 100:.2f}%")
    print(f"Decision Tree Precision: {dt_metrics.precision * 100:.2f}%")
    print(f"Decision Tree Recall   : {dt_metrics.recall * 100:.2f}%")
    print(f"Decision Tree F1-Score : {dt_metrics.f1 * 100:.2f}%")
    print("\nConfusion Matrix (DT):")
    print(f"  [TN={dt_metrics.confusion_matrix[0][0]}, FP={dt_metrics.confusion_matrix[0][1]}]")
    print(f"  [FN={dt_metrics.confusion_matrix[1][0]}, TP={dt_metrics.confusion_matrix[1][1]}]")
    print("\nClassification Report (DT):\n", dt_metrics.report)

    # 2. Random Forest Model
    print("--- 2. Training Random Forest Classifier ---")
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=args.random_state)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest Accuracy : {rf_metrics.accuracy * 100:.2f}%")
    print(f"Random Forest Precision: {rf_metrics.precision * 100:.2f}%")
    print(f"Random Forest Recall   : {rf_metrics.recall * 100:.2f}%")
    print(f"Random Forest F1-Score : {rf_metrics.f1 * 100:.2f}%")
    print("\nConfusion Matrix (RF):")
    print(f"  [TN={rf_metrics.confusion_matrix[0][0]}, FP={rf_metrics.confusion_matrix[0][1]}]")
    print(f"  [FN={rf_metrics.confusion_matrix[1][0]}, TP={rf_metrics.confusion_matrix[1][1]}]")
    print("\nClassification Report (RF):\n", rf_metrics.report)

    # Save models
    print("--- 3. Serializing and Exporting Models ---")
    save_model(dt_model, "decisiontree.joblib")
    print("Saved: decisiontree.joblib")

    # Also export pickle for backward compatibility with classifier.pkl
    with open("classifier.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    print("Saved: classifier.pkl")

    save_model(rf_model, "randomforest.joblib")
    print("Saved: randomforest.joblib")

    print("\nTraining and export completed successfully!")


if __name__ == "__main__":
    main()

