import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
HEX_IPV4_RE = re.compile(r"0x[0-9a-fA-F]{1,2}")

SUSPICIOUS_WORDS = {
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "password", "bank", "billing", "invoice",
    "payment", "pay", "alert", "support", "helpdesk", "reset", "unlock",
    "suspended", "limited", "security"
}

COMMON_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "t.ly", "lnkd.in", "s.id", "rb.gy"
}


def shannon_entropy(s: str) -> float:
    """Measure randomness: phishing URLs often contain high-entropy strings."""
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * np.log2(p)
    return float(ent)


def normalize_url(url: str) -> str:
    """Ensure urlparse works even if the scheme is missing."""
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    return u


def looks_like_ip(hostname: str) -> int:
    """Detect literal IPv4 hostnames like 192.168.0.1."""
    if not hostname:
        return 0
    return int(bool(IPV4_RE.match(hostname)))


def has_hex_ip(url: str) -> int:
    """Detect hex-encoded IP segments like 0xC0.0xA8..."""
    if not url:
        return 0
    return int(bool(HEX_IPV4_RE.search(url)))


def extract_url_features(url: str) -> dict:
    """
    Extract explainable numeric features from a URL string.

    Output is suitable for scikit-learn models (e.g., logistic regression).
    """
    u = normalize_url(url)
    try:
        parsed = urlparse(u)
    except ValueError:
        # Some strings look like URLs but contain malformed IPv6 (e.g., bad brackets)
        # Try a light sanitization; if still invalid, fall back to minimal parsing.
        u_sanitized = u.replace("[", "").replace("]", "")
        try:
            parsed = urlparse(u_sanitized)
            u = u_sanitized
        except ValueError:
            # Hard fallback: treat the whole thing as a path-like string
            parsed = urlparse("http://invalid.local/")
            hostname = ""
            path = u
            query = ""
            scheme = ""
            # Continue with features using these fallback values

    # If we entered the hard fallback above, scheme/hostname/path/query may already be set.
    # Otherwise, extract them normally from parsed.
    if "scheme" not in locals():
        scheme = (parsed.scheme or "").lower()
    if "hostname" not in locals():
        hostname = (parsed.hostname or "").lower()
    if "path" not in locals():
        path = parsed.path or ""
    if "query" not in locals():
        query = parsed.query or ""

    url_len = len(u)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    digit_count = sum(ch.isdigit() for ch in u)
    letter_count = sum(ch.isalpha() for ch in u)
    special_count = url_len - digit_count - letter_count

    dot_count = hostname.count(".") if hostname else 0
    hyphen_count = hostname.count("-") if hostname else 0
    underscore_count = u.count("_")
    slash_count = u.count("/")
    at_count = u.count("@")
    amp_count = u.count("&")
    eq_count = u.count("=")
    percent_count = u.count("%")

    host_parts = [p for p in hostname.split(".") if p] if hostname else []
    subdomain_parts = max(0, len(host_parts) - 2)
    tld = host_parts[-1] if len(host_parts) >= 2 else ""

    has_https = int(scheme == "https")
    has_ip = looks_like_ip(hostname)
    hex_ip = has_hex_ip(u)
    is_shortener = int(hostname in COMMON_SHORTENERS)

    token_blob = f"{hostname} {path} {query}".lower()
    keyword_hits = sum(1 for w in SUSPICIOUS_WORDS if w in token_blob)

    host_entropy = shannon_entropy(hostname)
    path_entropy = shannon_entropy(path)

    has_double_slash_in_path = int("//" in path)
    has_www = int(hostname.startswith("www."))
    suspicious_tld = int(tld in {"tk", "ml", "ga", "cf", "gq"})  # heuristic

    return {
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "query_len": query_len,
        "digit_count": digit_count,
        "special_count": special_count,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "underscore_count": underscore_count,
        "slash_count": slash_count,
        "at_count": at_count,
        "amp_count": amp_count,
        "eq_count": eq_count,
        "percent_count": percent_count,
        "subdomain_parts": subdomain_parts,
        "has_https": has_https,
        "has_ip": has_ip,
        "has_hex_ip": hex_ip,
        "is_shortener": is_shortener,
        "keyword_hits": keyword_hits,
        "host_entropy": host_entropy,
        "path_entropy": path_entropy,
        "has_double_slash_in_path": has_double_slash_in_path,
        "has_www": has_www,
        "suspicious_tld": suspicious_tld,
    }


def urls_to_feature_df(url_series: pd.Series) -> pd.DataFrame:
    """Convert a pandas Series of URLs into a DataFrame of features."""
    rows = [extract_url_features(u) for u in url_series.fillna("")]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # demo
    samples = pd.Series([
        "https://accounts.google.com/signin/v2/identifier",
        "http://192.168.0.1/login.php?session=123",
        "bit.ly/3xyzABC",
        "secure-paypal-verification.tk/account/update?user=abc",
        "example.com"
    ])

    df = urls_to_feature_df(samples)
    print(df)