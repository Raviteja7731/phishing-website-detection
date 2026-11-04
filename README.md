# 🛡️ Phishing Website Detection using Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning security system designed to detect and classify phishing websites and malicious URLs using structural, lexical, and domain indicators.

Phishing attacks trick users into sharing sensitive credentials by spoofing authentic brands, payment gateways, and banking institutions. This project performs **zero-payload threat analysis**—inspecting the structural signatures of URLs without requiring users to navigate to potentially infected endpoints.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Installation & Quickstart](#-installation--quickstart)
- [Usage Guide](#-usage-guide)
  - [1. Interactive Web Application](#1-interactive-web-application)
  - [2. Command-Line Prediction (CLI)](#2-command-line-prediction-cli)
  - [3. Model Training & Export](#3-model-training--export)
  - [4. Running Automated Tests](#4-running-automated-tests)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🔬 Overview & Architecture

```
                    +---------------------------+
                    |        Target URL         |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Feature Extraction Engine |
                    |   (13 Security Features)  |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Machine Learning Classifer|
                    | (Decision Tree / RF / Ens)|
                    +-------------+-------------+
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
       ✅ Legitimate Site                    🚨 Phishing Website
       - Safety Indicators Verified          - Flagged Risk Factors
       - Safe Domain Hierarchy               - High Risk Score
```

---

## ✨ Key Features

- **Real-Time URL Threat Detection**: Instant classification into **Legitimate** or **Phishing** categories.
- **Interpretable Risk Attribution**: Provides actionable security breakdowns explaining *why* a URL was flagged (e.g., brand-spoofing hyphens, masked `@` credentials, suspicious URL shorteners, excessive subdomain depth).
- **Interactive Streamlit Web Dashboard (`webapp.py`)**:
  - **Single URL Scanner**: Live prediction with visual risk meters, probability gauges, and feature inspection.
  - **Batch URL Scanner**: Bulk scan lists of URLs with CSV export capability.
  - **Model & Dataset Analytics**: View model performance metrics and feature importance charts.
  - **Educational Security Guide**: Quick-reference guide on identifying deceptive URLs.
- **Fast CLI Interface (`predict.py`)**: Predict individual URLs from the terminal with support for formatted or JSON output.
- **Modular Codebase**: Separation of concerns with dedicated `src/features.py`, `src/model.py`, and `src/predictor.py`.
- **Reproducible ML Pipeline (`train.py`)**: Train, evaluate, and serialize modern Scikit-Learn models with a single command.
- **Unit Tested**: Full test coverage across feature extraction routines and prediction pipelines using `pytest`.

---

## 📁 Project Structure

```
phishing-website-detection/
├── .gitignore                      # Git ignore rules for Python, virtualenv, and caches
├── README.md                       # Comprehensive project documentation
├── Procfile                        # Cloud application deployment configuration
├── setup.sh                        # Streamlit production configuration script
├── requirements.txt                # Python project dependencies
├── legitimate-urls.csv             # Labeled legitimate website feature dataset (1,017 samples)
├── phishing-urls.csv               # Labeled phishing website feature dataset (998 samples)
├── classifier.pkl                  # Serialized Decision Tree model (pickle)
├── decisiontree.joblib             # Serialized Decision Tree model (joblib)
├── randomforest.joblib             # Serialized Random Forest ensemble model (joblib)
├── webapp.py                       # Interactive Streamlit web application
├── train.py                        # Model training and evaluation CLI
├── predict.py                      # URL inspection and classification CLI
├── src/                            # Modular source code package
│   ├── __init__.py                 # Package initializer
│   ├── features.py                 # FeatureExtraction class and feature vector builder
│   ├── model.py                    # Dataset loading, training, and evaluation utilities
│   └── predictor.py                # PhishingPredictor inference engine and risk explainer
├── tests/                          # Automated unit test suite
│   ├── __init__.py
│   ├── test_features.py            # Unit tests for URL feature extraction
│   └── test_model.py               # Unit tests for model inference and datasets
└── notebooks/                      # Exploratory data science notebooks
    ├── phishing-website.ipynb      # Raw URL scraping and feature extraction notebook
    └── Training and Prediction.ipynb# Model experimentation and training notebook
```

---

## 📊 Dataset & Feature Engineering

The model is trained on a curated corpus of **2,015 websites** comprising:
- **1,017 Legitimate URLs** (labeled `0`) sourced from popular global directories.
- **998 Phishing URLs** (labeled `1`) extracted from real-world threat feeds.

### The 13 Predictive Features

| # | Feature Name | Description | Values / Encoding |
|---|--------------|-------------|-------------------|
| 1 | `Having_IP` | Checks if the host component is a raw IPv4/IPv6 address rather than a registered domain name. | `0` = Domain Name, `1` = IP Address |
| 2 | `URL_Length` | Evaluates URL string length. Phishers use long URLs to hide redirect tokens and parameters. | `0` = <54 chars (Safe), `2` = 54-75 chars (Suspicious), `1` = >75 chars (Phishing) |
| 3 | `Having_@_symbol` | Checks for the `@` character, which causes browsers to ignore all preceding authority information. | `0` = Absent, `1` = Present |
| 4 | `Redirection_//_symbol` | Checks if `//` appears inside the URL path, often used to bypass path validation. | `0` = Absent, `1` = Present |
| 5 | `Prefix_suffix_separation` | Checks if hyphens `-` appear in the domain name to spoof legitimate company brand names. | `0` = Absent, `1` = Present |
| 6 | `Sub_domains` | Counts dot `.` delimiters in the domain. Excessive subdomains disguise the real base domain. | `0` = <3 dots, `2` = 3 dots, `1` = >3 dots |
| 7 | `tiny_url` | Checks if the link uses URL shortening services (`bit.ly`, `tinyurl`, `goo.gl`, etc.). | `0` = Direct URL, `1` = Shortened URL |
| 8 | `http_tokens` | Checks if `http` or `https` tokens are embedded in the domain or path to mislead users. | `0` = Normal, `1` = Spoofed Token |
| 9 | `dns_record` | Verifies whether the domain has an active, valid DNS or WHOIS registry record. | `0` = Valid, `1` = No Record / Failed |
| 10 | `web_traffic` | Evaluates domain popularity and traffic volume. Phishing pages lack natural web traffic. | `0` = High Traffic, `1` = Unranked, `2` = Moderate |
| 11 | `domain_registration_length` | Measures remaining domain registration validity. Phishing domains are rarely registered for >1 year. | `0` = >1 Year, `1` = ≤1 Year |
| 12 | `age_domain` | Evaluates domain age. Ephemeral domains (<6 months) present significantly higher risk. | `0` = ≥6 Months, `1` = <6 Months |
| 13 | `statistical_report` | Checks if the host matches known threat lists or suspicious autonomous systems. | `0` = Clean, `1` = Flagged Host |

---

## 📈 Model Performance & Evaluation

Both models were trained using an 80/20 stratified split (`1,612` training samples, `403` test samples):

| Metric | Decision Tree Classifier | Random Forest Classifier |
|---|:---:|:---:|
| **Accuracy** | **81.39%** | **81.39%** |
| **Precision (Phishing)** | **85.71%** | **85.31%** |
| **Recall (Phishing)** | **75.00%** | **75.50%** |
| **F1-Score (Phishing)** | **80.00%** | **80.11%** |
| **True Negatives (TN)** | 178 | 177 |
| **False Positives (FP)** | 25 | 26 |
| **False Negatives (FN)** | 50 | 49 |
| **True Positives (TP)** | 150 | 151 |

### Top 5 Predictive Features by Importance

1. **`URL_Length`** (21.6%)
2. **`web_traffic`** (19.3%)
3. **`statistical_report`** (14.2%)
4. **`dns_record`** (8.8%)
5. **`age_domain`** (8.4%)

---

## 🚀 Installation & Quickstart

### Prerequisites

- Python 3.9, 3.10, 3.11, 3.12, or 3.14+
- `git`

### Step 1: Clone the Repository

```bash
git clone https://github.com/Raviteja7731/phishing-website-detection.git
cd phishing-website-detection
```

### Step 2: Create and Activate a Virtual Environment

```bash
# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage Guide

### 1. Interactive Web Application

Launch the Streamlit web dashboard locally:

```bash
streamlit run webapp.py
```

The application will open automatically at `http://localhost:8501`.

Features included in the web interface:
- **Single URL Inspection**: Test any URL, view the verdict (Legitimate vs Phishing), confidence percentage, and full feature breakdown.
- **Batch Scanner**: Paste a list of URLs to scan multiple sites at once and export the report to CSV.
- **Analytics View**: Visual charts showing model metrics and feature importance.
- **Security Knowledge Base**: Educational tips on spotting phishing schemes.

---

### 2. Command-Line Prediction (CLI)

Analyze any URL directly from your terminal:

```bash
# Analyze a legitimate URL
python predict.py "https://www.google.com"

# Analyze a suspicious URL
python predict.py "http://192.168.1.100/paypal-update/account-login-verification/secure-session/index.php"

# Output structured JSON for automation / CI pipelines
python predict.py "https://github.com" --json
```

**Options:**
- `--model <path>`: Specify a custom model file (default: `decisiontree.joblib`).
- `--enable-network`: Enable live network lookups for WHOIS/DNS queries.
- `--json`: Output raw JSON results instead of human-readable text.

---

### 3. Model Training & Export

Retrain models from the raw dataset and re-export serialized weights:

```bash
python train.py
```

**Custom training arguments:**
```bash
python train.py --legit legitimate-urls.csv --phish phishing-urls.csv --test-size 0.25 --random-state 100
```

This generates updated `decisiontree.joblib`, `randomforest.joblib`, and `classifier.pkl` files.

---

### 4. Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

All 14 unit tests cover feature extraction rules, URL edge cases, model deserialization, and prediction logic.

---

## ☁️ Deployment

### Streamlit Cloud / Render / Heroku

The repository includes production-ready deployment configurations:
- **`Procfile`**: `web: sh setup.sh && streamlit run webapp.py`
- **`setup.sh`**: Configures headless mode, port binding, and disables CORS.

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Connect your GitHub repository.
2. Set the main file path to `webapp.py`.
3. Click **Deploy**.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
