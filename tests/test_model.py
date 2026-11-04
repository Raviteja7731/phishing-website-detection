"""
Tests for model loading, evaluation, and prediction pipeline.
"""

import os
import pytest
from src.features import FEATURE_NAMES
from src.model import load_dataset, load_model
from src.predictor import PhishingPredictor


def test_model_files_exist():
    assert os.path.exists("decisiontree.joblib"), "decisiontree.joblib must exist"
    assert os.path.exists("classifier.pkl"), "classifier.pkl must exist"
    assert os.path.exists("randomforest.joblib"), "randomforest.joblib must exist"


def test_dataset_loading():
    X_train, X_test, y_train, y_test = load_dataset()
    assert len(X_train.columns) == 13
    assert list(X_train.columns) == FEATURE_NAMES
    assert len(X_train) + len(X_test) == 2015


def test_model_loading():
    dt = load_model("decisiontree.joblib")
    assert hasattr(dt, "predict")

    rf = load_model("randomforest.joblib")
    assert hasattr(rf, "predict")


def test_predictor_legitimate():
    predictor = PhishingPredictor(model_path="decisiontree.joblib", enable_network_lookups=False)
    res = predictor.predict("https://www.google.com")

    assert res.prediction in ["Legitimate", "Phishing"]
    assert res.is_phishing is False
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.features) == 13


def test_predictor_phishing_ip():
    predictor = PhishingPredictor(model_path="decisiontree.joblib", enable_network_lookups=False)
    # Long URL with IP and deep subdomains
    res = predictor.predict(
        "http://192.168.1.100/paypal-update/account-login-verification/secure-session/index.php"
    )

    assert res.prediction in ["Legitimate", "Phishing"]
    assert res.is_phishing is True
    assert any("IP address" in rf for rf in res.risk_factors)

