"""
Streamlit Web Application for Phishing Website Detection.
Provides interactive single-URL scanning, batch URL inspection, model metrics,
and feature explanation.
"""

import pandas as pd
import streamlit as st
from src.features import FEATURE_NAMES, FeatureExtraction
from src.model import load_model
from src.predictor import PhishingPredictor

# Page Configuration
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .risk-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor(model_choice: str, enable_network: bool):
    model_file = "randomforest.joblib" if model_choice == "Random Forest" else "decisiontree.joblib"
    return PhishingPredictor(model_path=model_file, enable_network_lookups=enable_network)


def main():
    st.sidebar.title("🛡️ Controls & Config")
    st.sidebar.markdown("Configure inference model and detection settings:")

    model_choice = st.sidebar.selectbox(
        "Select Machine Learning Model",
        options=["Decision Tree", "Random Forest"],
        index=0,
        help="Choose classifier for inference.",
    )

    enable_network = st.sidebar.checkbox(
        "Enable Live Network Checks (WHOIS/DNS)",
        value=False,
        help="When enabled, queries live DNS and WHOIS servers. Keep disabled for instant offline evaluation.",
    )

    predictor = get_predictor(model_choice, enable_network)

    # Sidebar Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🔍 Single URL Scanner", "📋 Batch URL Scanner", "📊 Model & Dataset Analytics", "ℹ️ About & Features Guide"],
    )

    if page == "🔍 Single URL Scanner":
        render_single_scanner(predictor)
    elif page == "📋 Batch URL Scanner":
        render_batch_scanner(predictor)
    elif page == "📊 Model & Dataset Analytics":
        render_analytics()
    elif page == "ℹ️ About & Features Guide":
        render_about_guide()


def render_single_scanner(predictor: PhishingPredictor):
    st.markdown('<div class="main-header">🛡️ Phishing Website Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Analyze and classify URLs in real-time using machine learning pattern extraction.</div>',
        unsafe_allow_html=True,
    )

    # Preset examples for convenience
    preset_urls = {
        "Custom URL": "",
        "Legitimate: Google Search": "https://www.google.com",
        "Legitimate: GitHub Repository": "https://github.com/Raviteja7731/phishing-website-detection",
        "Phishing: IP address host": "http://192.168.1.100/paypal-update/account-login-verification/secure-session/index.php",
        "Phishing: Shortened Suspicious URL": "http://bit.ly/bank-security-urgent-login",
        "Phishing: Subdomain Spoofing": "http://paypal.com.account-verify.fake-phish.net/login/confirm",
    }

    selected_preset = st.selectbox("Select a quick sample or enter your own:", list(preset_urls.keys()))

    default_value = preset_urls[selected_preset] if selected_preset != "Custom URL" else ""

    url_input = st.text_input(
        "Enter target URL to inspect:",
        value=default_value,
        placeholder="https://example.com/login",
    )

    if st.button("Inspect URL", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
            return

        with st.spinner("Analyzing URL structure and extracting security features..."):
            res = predictor.predict(url_input)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            if res.is_phishing:
                st.error(f"### 🚨 Verdict: PHISHING DETECTED")
                st.markdown(
                    f"**Confidence**: `{res.confidence * 100:.1f}%`  \n"
                    f"**Target**: `{res.url}`  \n"
                    f"This website demonstrates behavioral and structural patterns strongly indicative of malicious phishing."
                )
            else:
                st.success(f"### ✅ Verdict: LEGITIMATE WEBSITE")
                st.markdown(
                    f"**Confidence**: `{res.confidence * 100:.1f}%`  \n"
                    f"**Target**: `{res.url}`  \n"
                    f"This website structure matches standard legitimate domain patterns."
                )

            st.write("#### Probability Distribution")
            legit_prob = res.probabilities.get("Legitimate", 0.0)
            phish_prob = res.probabilities.get("Phishing", 0.0)

            st.markdown(f"**Legitimate**: {legit_prob * 100:.1f}%")
            st.progress(min(max(legit_prob, 0.0), 1.0))

            st.markdown(f"**Phishing**: {phish_prob * 100:.1f}%")
            st.progress(min(max(phish_prob, 0.0), 1.0))

        with col2:
            st.write("#### 🛡️ Risk & Safety Analysis")
            if res.risk_factors:
                st.markdown("**⚠️ Identified Risk Indicators:**")
                for rf in res.risk_factors:
                    st.markdown(f"- 🔴 {rf}")
            else:
                st.markdown("✅ No critical structural risks flagged.")

            if res.safe_factors:
                st.markdown("**🛡️ Verified Safe Indicators:**")
                for sf in res.safe_factors:
                    st.markdown(f"- 🟢 {sf}")

        st.markdown("---")
        st.write("#### 🔬 Detailed Feature Vector")
        feature_data = []
        for feat, val in res.features.items():
            status_desc = "Safe (0)" if val == 0 else ("Phishing (1)" if val == 1 else "Suspicious (2)")
            feature_data.append({"Feature Name": feat, "Value": val, "Evaluation": status_desc})

        feat_df = pd.DataFrame(feature_data)
        st.dataframe(feat_df, use_container_width=True)


def render_batch_scanner(predictor: PhishingPredictor):
    st.markdown('<div class="main-header">📋 Batch URL Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Process multiple URLs simultaneously and download detailed risk reports.</div>',
        unsafe_allow_html=True,
    )

    raw_urls = st.text_area(
        "Enter multiple URLs (one per line):",
        height=150,
        placeholder="https://google.com\nhttp://192.168.1.1/login\nhttps://github.com\nhttp://bit.ly/secure-account",
    )

    if st.button("Scan Batch", type="primary"):
        urls = [u.strip() for u in raw_urls.split("\n") if u.strip()]
        if not urls:
            st.warning("Please provide at least one URL.")
            return

        results = []
        progress_bar = st.progress(0)
        for i, url in enumerate(urls):
            res = predictor.predict(url)
            row = {
                "URL": res.url,
                "Verdict": res.prediction,
                "Confidence": f"{res.confidence * 100:.1f}%",
                "Phishing Prob": f"{res.probabilities.get('Phishing', 0.0) * 100:.1f}%",
                "Legitimate Prob": f"{res.probabilities.get('Legitimate', 0.0) * 100:.1f}%",
                "Risk Count": len(res.risk_factors),
            }
            results.append(row)
            progress_bar.progress((i + 1) / len(urls))

        df_results = pd.DataFrame(results)
        st.success(f"Successfully evaluated {len(urls)} URLs!")
        st.dataframe(df_results, use_container_width=True)

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="phishing_batch_scan_results.csv",
            mime="text/csv",
        )


def render_analytics():
    st.markdown('<div class="main-header">📊 Model & Dataset Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluation metrics, dataset distribution, and feature importance rankings.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total URLs Analyzed", "2,015")
    with c2:
        st.metric("Legitimate Samples", "1,017 (50.5%)")
    with c3:
        st.metric("Phishing Samples", "998 (49.5%)")
    with c4:
        st.metric("Model Test Accuracy", "81.4%")

    st.markdown("---")

    st.write("#### 📈 Model Comparison")
    perf_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Decision Tree": ["81.39%", "85.71%", "75.00%", "80.00%"],
        "Random Forest": ["81.39%", "85.31%", "75.50%", "80.11%"],
    }
    st.table(pd.DataFrame(perf_data))

    st.write("#### 🎯 Feature Importance Ranking")
    features_importance = {
        "Feature": [
            "URL_Length",
            "web_traffic",
            "statistical_report",
            "dns_record",
            "age_domain",
            "Sub_domains",
            "domain_registration_length",
            "Prefix_suffix_separation",
            "tiny_url",
            "Having_IP",
            "Having_@_symbol",
            "Redirection_//_symbol",
            "http_tokens",
        ],
        "Importance (%)": [21.6, 19.3, 14.2, 8.8, 8.4, 8.0, 6.6, 5.6, 5.4, 0.8, 0.6, 0.6, 0.1],
    }
    st.bar_chart(data=pd.DataFrame(features_importance).set_index("Feature"))


def render_about_guide():
    st.markdown('<div class="main-header">ℹ️ Features & Security Guide</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### How Machine Learning Detects Phishing Websites
        
        Phishing websites deceive users by mimicking trusted institutions (banks, payment gateways, social networks).
        This system extracts **13 structural, lexical, and domain indicators** from any URL without downloading dangerous payloads.
        
        #### 13 Key Extracted Features:
        1. **Having_IP**: Legitimate domains use domain names. Attackers often use direct IP addresses (e.g. `http://192.168.1.1/login`).
        2. **URL_Length**: Phishing URLs are frequently lengthened (> 75 chars) to obfuscate malicious path tokens.
        3. **tiny_url**: Attackers abuse URL shorteners (`bit.ly`, `tinyurl.com`, etc.) to hide suspicious domains.
        4. **Having_@_symbol**: Browsers treat text before `@` as user credentials, masking the true host.
        5. **Redirection_//_symbol**: Checks if double slashes `//` appear in the path to trick browser navigation.
        6. **Prefix_suffix_separation**: Use of hyphens `-` in domains (e.g., `paypal-login.com`) to spoof brands.
        7. **Sub_domains**: Multiple dot separators indicate deep subdomains created to disguise the true root domain.
        8. **dns_record**: Legitimate sites maintain resolvable DNS and WHOIS registry records.
        9. **web_traffic**: High-ranking websites are overwhelmingly legitimate; brand new phishing pages lack search engine indexation and traffic.
        10. **domain_registration_length**: Phishing domains are ephemeral, rarely registered for more than 1 year.
        11. **age_domain**: Newly registered domains (< 6 months old) carry significantly higher risk.
        12. **statistical_report**: Checks against known malicious hosting autonomous systems and phishing threat lists.
        13. **http_tokens**: Insertion of `http` or `https` inside domain or path to deceive casual users.
        """
    )


if __name__ == "__main__":
    main()

