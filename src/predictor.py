"""
Inference and risk assessment pipeline for Phishing Website Detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .features import FEATURE_NAMES, FeatureExtraction
from .model import load_model


@dataclass
class PredictionResult:
    url: str
    prediction: str  # "Legitimate" or "Phishing"
    is_phishing: bool
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, int]
    risk_factors: List[str]
    safe_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "prediction": self.prediction,
            "is_phishing": self.is_phishing,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "features": self.features,
            "risk_factors": self.risk_factors,
            "safe_factors": self.safe_factors,
        }


class PhishingPredictor:
    """Predicts whether a URL is phishing or legitimate and provides explanations."""

    def __init__(
        self,
        model_path: str = "decisiontree.joblib",
        enable_network_lookups: bool = False,
    ):
        self.model_path = model_path
        self.model = load_model(model_path)
        self.extractor = FeatureExtraction(
            enable_network_lookups=enable_network_lookups, timeout=2.0
        )

    def predict(self, url: str) -> PredictionResult:
        """Run full prediction and risk explanation on a URL."""
        features = self.extractor.extract_features(url)
        df_features = pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

        pred_class = int(self.model.predict(df_features)[0])
        is_phishing = bool(pred_class == 1)
        prediction_label = "Phishing" if is_phishing else "Legitimate"

        probabilities: Dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_features)[0]
            classes = list(getattr(self.model, "classes_", [0, 1]))
            legit_idx = classes.index(0) if 0 in classes else 0
            phish_idx = classes.index(1) if 1 in classes else 1
            probabilities["Legitimate"] = float(probs[legit_idx])
            probabilities["Phishing"] = float(probs[phish_idx])
            confidence = probabilities[prediction_label]
        else:
            confidence = 1.0
            probabilities["Legitimate"] = 0.0 if is_phishing else 1.0
            probabilities["Phishing"] = 1.0 if is_phishing else 0.0

        risk_factors = []
        safe_factors = []

        if features["Having_IP"] == 1:
            risk_factors.append("URL domain uses a raw IP address instead of a domain name.")
        else:
            safe_factors.append("Standard domain name format used (not a raw IP).")

        if features["Having_@_symbol"] == 1:
            risk_factors.append("Contains '@' symbol, which can mask the true destination.")
        else:
            safe_factors.append("No credential-masking '@' symbol.")

        if features["tiny_url"] == 1:
            risk_factors.append("Uses a URL shortening service hiding the real target destination.")
        else:
            safe_factors.append("Direct destination URL (not shortened).")

        if features["URL_Length"] == 1:
            risk_factors.append("Unusually long URL (> 75 characters) often used to hide parameters.")
        elif features["URL_Length"] == 2:
            risk_factors.append("Moderately long URL (54-75 characters).")
        else:
            safe_factors.append("Standard concise URL length (< 54 characters).")

        if features["Prefix_suffix_separation"] == 1:
            risk_factors.append("Domain contains hyphen '-' (frequently used in domain impersonation).")
        else:
            safe_factors.append("No suspicious hyphen separation in domain.")

        if features["Sub_domains"] == 1:
            risk_factors.append("Deep subdomain structure (> 3 dots) common in deceptive subdomains.")
        elif features["Sub_domains"] == 2:
            risk_factors.append("Suspicious subdomain count (3 dots).")
        else:
            safe_factors.append("Clean domain hierarchy (< 3 dots).")

        if features["Redirection_//_symbol"] == 1:
            risk_factors.append("Contains '//' inside path component indicating redirection trick.")

        if features["http_tokens"] == 1:
            risk_factors.append("Token 'http/https' embedded inside host or path.")

        if features["statistical_report"] == 1:
            risk_factors.append("Matched known suspicious domains or malicious hosting ranges.")

        return PredictionResult(
            url=url,
            prediction=prediction_label,
            is_phishing=is_phishing,
            confidence=confidence,
            probabilities=probabilities,
            features=features,
            risk_factors=risk_factors,
            safe_factors=safe_factors,
        )

