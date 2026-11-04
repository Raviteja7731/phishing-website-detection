"""
Feature extraction module for Phishing Website Detection.

Extracts lexical, domain, and security indicator features from URLs
matching the schema used to train the machine learning models.
"""

from __future__ import annotations

import re
import socket
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

try:
    import whois
except ImportError:
    whois = None


# Feature column order expected by the trained models
FEATURE_NAMES: List[str] = [
    "Having_@_symbol",
    "Having_IP",
    "Prefix_suffix_separation",
    "Redirection_//_symbol",
    "Sub_domains",
    "URL_Length",
    "age_domain",
    "dns_record",
    "domain_registration_length",
    "http_tokens",
    "statistical_report",
    "tiny_url",
    "web_traffic",
]

SHORTENING_SERVICES_REGEX = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|"
    r"db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|"
    r"q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net",
    re.IGNORECASE,
)

SUSPICIOUS_DOMAINS_REGEX = re.compile(
    r"at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly",
    re.IGNORECASE,
)

KNOWN_MALICIOUS_IPS_REGEX = re.compile(
    r"146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|"
    r"181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|"
    r"107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|"
    r"107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|"
    r"118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|"
    r"141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|"
    r"216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|"
    r"213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|"
    r"34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|"
    r"198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|"
    r"209\.99\.17\.27|216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|54\.86\.225\.156|"
    r"54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42"
)


class FeatureExtraction:
    """Extracts features from URLs for machine learning phishing detection."""

    def __init__(self, enable_network_lookups: bool = True, timeout: float = 2.0):
        """
        Args:
            enable_network_lookups: Whether to perform DNS/WHOIS network queries.
                                   If False, uses fast offline heuristics.
            timeout: Maximum timeout in seconds for network queries.
        """
        self.enable_network_lookups = enable_network_lookups
        self.timeout = timeout

    @staticmethod
    def normalize_url(url: str) -> str:
        """Ensure URL has a scheme for reliable parsing."""
        url = url.strip()
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "http://" + url
        return url

    def get_protocol(self, url: str) -> str:
        """Extract URL scheme/protocol."""
        return urlparse(self.normalize_url(url)).scheme

    def get_domain(self, url: str) -> str:
        """Extract domain / network location."""
        return urlparse(self.normalize_url(url)).netloc

    def get_path(self, url: str) -> str:
        """Extract URL path."""
        return urlparse(self.normalize_url(url)).path

    def having_ip(self, url: str) -> int:
        """
        Check if the domain part contains an IP address (IPv4 or IPv6).
        Returns:
            1: IP address found (phishing indicator)
            0: Legitimate domain name
        """
        ip_pattern = (
            r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\."
            r"([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])/?)|"
            r"((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})/?)|"
            r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}"
        )
        return 1 if re.search(ip_pattern, url) else 0

    def url_length(self, url: str) -> int:
        """
        Classify URL length.
        Returns:
            0: < 54 characters (legitimate)
            2: 54 to 75 characters (suspicious)
            1: > 75 characters (phishing)
        """
        length = len(url)
        if length < 54:
            return 0
        elif 54 <= length <= 75:
            return 2
        return 1

    def having_at_symbol(self, url: str) -> int:
        """Check for presence of '@' symbol which masks preceding URL credentials."""
        return 1 if "@" in url else 0

    def redirection(self, url: str) -> int:
        """Check if '//' appears in the path component (redirection trick)."""
        norm_url = self.normalize_url(url)
        path = urlparse(norm_url).path
        return 1 if "//" in path else 0

    def prefix_suffix_separation(self, url: str) -> int:
        """Check if '-' appears in the domain name to spoof brand prefixes."""
        domain = self.get_domain(url)
        return 1 if "-" in domain else 0

    def sub_domains(self, url: str) -> int:
        """
        Classify subdomain depth based on dot count in domain/url.
        Returns:
            0: < 3 dots (legitimate)
            2: exactly 3 dots (suspicious)
            1: > 3 dots (phishing)
        """
        dot_count = url.count(".")
        if dot_count < 3:
            return 0
        elif dot_count == 3:
            return 2
        return 1

    def shortening_service(self, url: str) -> int:
        """Detect known URL shortening services."""
        return 1 if SHORTENING_SERVICES_REGEX.search(url) else 0

    def https_token(self, url: str) -> int:
        """
        Check if 'http' or 'https' token appears inside the host or path after protocol.
        Example: http://paypal.com.https-account.com
        """
        norm = self.normalize_url(url)
        match = re.search(r"^https?://", norm, re.IGNORECASE)
        if match:
            rest = norm[match.end():]
            return 1 if re.search(r"https?", rest, re.IGNORECASE) else 0
        return 0

    def statistical_report(self, url: str) -> int:
        """
        Check URL against known malicious domains and IP addresses.
        """
        if SUSPICIOUS_DOMAINS_REGEX.search(url):
            return 1

        if not self.enable_network_lookups:
            return 0

        domain = self.get_domain(url)
        hostname = domain.split(":")[0]
        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.timeout)
            try:
                ip_address = socket.gethostbyname(hostname)
                if KNOWN_MALICIOUS_IPS_REGEX.search(ip_address):
                    return 1
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception:
            pass
        return 0

    def _query_whois(self, domain: str):
        """Query WHOIS record with timeout safety."""
        if not self.enable_network_lookups or whois is None:
            return None
        try:
            return whois.whois(domain)
        except Exception:
            return None

    def dns_record(self, url: str) -> int:
        """
        Check DNS / WHOIS record existence.
        Returns:
            0: Valid DNS/domain record exists
            1: No record found or lookup failed (phishing indicator)
        """
        domain = self.get_domain(url).split(":")[0]
        if not domain:
            return 1

        if not self.enable_network_lookups:
            return 0

        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.timeout)
            try:
                socket.gethostbyname(domain)
                return 0
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception:
            pass

        record = self._query_whois(domain)
        return 0 if record and getattr(record, "domain_name", None) else 1

    def domain_registration_length(self, url: str) -> int:
        """
        Check domain registration length.
        Phishing domains are typically registered for 1 year or less.
        Returns:
            0: Registration > 1 year (legitimate)
            1: <= 1 year or unresolvable (phishing indicator)
            2: Ambiguous expiration date
        """
        domain = self.get_domain(url).split(":")[0]
        record = self._query_whois(domain)
        if not record or not hasattr(record, "expiration_date") or record.expiration_date is None:
            return 1

        exp = record.expiration_date
        if isinstance(exp, list):
            exp = exp[0] if exp else None

        if not isinstance(exp, datetime):
            return 2

        try:
            days_left = (exp - datetime.now()).days
            return 0 if (days_left / 365) > 1 else 1
        except Exception:
            return 1

    def age_domain(self, url: str) -> int:
        """
        Check domain age. Phishing domains are often brand new (< 6 months).
        Returns:
            0: Age >= 6 months (legitimate)
            1: Age < 6 months or unresolvable (phishing indicator)
            2: Ambiguous creation/expiration dates
        """
        domain = self.get_domain(url).split(":")[0]
        record = self._query_whois(domain)
        if not record:
            return 1

        creation = getattr(record, "creation_date", None)
        expiration = getattr(record, "expiration_date", None)

        if isinstance(creation, list):
            creation = creation[0] if creation else None
        if isinstance(expiration, list):
            expiration = expiration[0] if expiration else None

        if not isinstance(creation, datetime):
            return 1

        try:
            today = datetime.now()
            age_days = abs((today - creation).days)
            return 0 if (age_days / 30) >= 6 else 1
        except Exception:
            return 1

    def web_traffic(self, url: str) -> int:
        """
        Evaluate website traffic / ranking.
        Note: The legacy Alexa API was sunsetted. When unavailable, returns:
            0: Popular / known legitimate high-traffic domain
            1: Unranked / unknown (suspicious/phishing indicator)
            2: Moderate ranking
        """
        domain = self.get_domain(url).lower()
        # Top known global websites for instant high-traffic classification
        top_sites = {
            "google.com", "youtube.com", "facebook.com", "wikipedia.org", "yahoo.com",
            "amazon.com", "twitter.com", "instagram.com", "linkedin.com", "reddit.com",
            "netflix.com", "microsoft.com", "apple.com", "github.com", "stackoverflow.com",
        }
        for site in top_sites:
            if domain == site or domain.endswith("." + site):
                return 0
        return 1

    def extract_features(self, url: str) -> Dict[str, int]:
        """
        Extract the complete 13-feature dictionary for a given URL.
        Features match the schema and column names expected by the models.
        """
        return {
            "Having_@_symbol": self.having_at_symbol(url),
            "Having_IP": self.having_ip(url),
            "Prefix_suffix_separation": self.prefix_suffix_separation(url),
            "Redirection_//_symbol": self.redirection(url),
            "Sub_domains": self.sub_domains(url),
            "URL_Length": self.url_length(url),
            "age_domain": self.age_domain(url),
            "dns_record": self.dns_record(url),
            "domain_registration_length": self.domain_registration_length(url),
            "http_tokens": self.https_token(url),
            "statistical_report": self.statistical_report(url),
            "tiny_url": self.shortening_service(url),
            "web_traffic": self.web_traffic(url),
        }

    def extract_features_vector(self, url: str) -> List[int]:
        """Extract features as an ordered list matching FEATURE_NAMES."""
        features_dict = self.extract_features(url)
        return [features_dict[name] for name in FEATURE_NAMES]
