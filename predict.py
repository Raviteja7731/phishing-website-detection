#!/usr/bin/env python3
"""
CLI URL Prediction Script for Phishing Website Detection.

Takes a URL as input and outputs whether it is legitimate or phishing,
along with confidence score, extracted features, and risk indicators.
"""

import argparse
import json
import sys
from src.predictor import PhishingPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a URL is Legitimate or Phishing."
    )
    parser.add_argument("url", help="URL to analyze")
    parser.add_argument(
        "--model",
        default="decisiontree.joblib",
        help="Path to trained model file (default: decisiontree.joblib)",
    )
    parser.add_argument(
        "--enable-network",
        action="store_true",
        help="Enable live DNS/WHOIS network queries (default: offline mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON results",
    )
    args = parser.parse_args()

    try:
        predictor = PhishingPredictor(
            model_path=args.model,
            enable_network_lookups=args.enable_network,
        )
        res = predictor.predict(args.url)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(res.to_dict(), indent=2))
        return

    print("=" * 60)
    print("       PHISHING WEBSITE DETECTION ANALYSIS")
    print("=" * 60)
    print(f"Target URL   : {res.url}")
    print(f"Verdict      : {'🚨 PHISHING' if res.is_phishing else '✅ LEGITIMATE'}")
    print(f"Confidence   : {res.confidence * 100:.1f}%")
    print("Probabilities:")
    for label, prob in res.probabilities.items():
        print(f"  - {label:<11}: {prob * 100:.1f}%")

    if res.risk_factors:
        print("\n⚠️  Detected Risk Factors:")
        for rf in res.risk_factors:
            print(f"  - {rf}")

    if res.safe_factors:
        print("\n🛡️  Safe Indicators:")
        for sf in res.safe_factors:
            print(f"  - {sf}")

    print("\nExtracted Feature Vector:")
    for feat_name, val in res.features.items():
        print(f"  {feat_name:<28}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
