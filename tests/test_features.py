"""
Tests for URL feature extraction logic.
"""

from src.features import FEATURE_NAMES, FeatureExtraction


def test_feature_names_length():
    assert len(FEATURE_NAMES) == 13
    assert "Having_IP" in FEATURE_NAMES
    assert "URL_Length" in FEATURE_NAMES
    assert "tiny_url" in FEATURE_NAMES


def test_having_ip():
    fe = FeatureExtraction(enable_network_lookups=False)
    assert fe.having_ip("http://192.168.1.1/login") == 1
    assert fe.having_ip("http://10.0.0.1/verify") == 1
    assert fe.having_ip("https://www.google.com") == 0
    assert fe.having_ip("https://github.com/login") == 0


def test_url_length():
    fe = FeatureExtraction(enable_network_lookups=False)
    short_url = "https://short.com"
    medium_url = "https://example.com/very-medium-length-url-sample-for-test"
    long_url = "https://very-long-suspicious-domain-name-trying-to-mimic-legitimate-bank-service-login.com/secure/session/redirect"

    assert fe.url_length(short_url) == 0
    assert fe.url_length(medium_url) == 2
    assert fe.url_length(long_url) == 1


def test_having_at_symbol():
    fe = FeatureExtraction(enable_network_lookups=False)
    assert fe.having_at_symbol("https://user@phishingsite.com") == 1
    assert fe.having_at_symbol("https://legitimatesite.com") == 0


def test_redirection():
    fe = FeatureExtraction(enable_network_lookups=False)
    assert fe.redirection("http://example.com/path//redirect") == 1
    assert fe.redirection("https://example.com/normal/path") == 0


def test_prefix_suffix_separation():
    fe = FeatureExtraction(enable_network_lookups=False)
    assert fe.prefix_suffix_separation("https://paypal-verify.com/login") == 1
    assert fe.prefix_suffix_separation("https://paypal.com/login") == 0


def test_sub_domains():
    fe = FeatureExtraction(enable_network_lookups=False)
    assert fe.sub_domains("https://google.com") == 0
    assert fe.sub_domains("https://sub.domain.co.uk") == 2
    assert fe.sub_domains("https://a.b.c.d.phish.com") == 1


def test_shortening_service():
    fe = FeatureExtraction(enable_network_lookups=False)
    assert fe.shortening_service("http://bit.ly/3xZyW") == 1
    assert fe.shortening_service("http://tinyurl.com/fake-login") == 1
    assert fe.shortening_service("https://google.com/search") == 0


def test_extract_features_vector():
    fe = FeatureExtraction(enable_network_lookups=False)
    vec = fe.extract_features_vector("https://www.google.com")
    assert len(vec) == 13
    assert all(isinstance(v, int) for v in vec)
