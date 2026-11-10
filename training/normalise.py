"""Text normalisation shared by training and the API. The browser has a line-for-line JavaScript copy."""
import re


def normalise(t):
    """Same rules as normalise() in the browser. Order matters."""
    t = t.lower()
    t = re.sub(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|in|co\.uk|biz|info|ly)\b\S*)", " _url_ ", t)
    t = re.sub(r"(£|\$|₹|rs\.?|inr)\s?\d[\d,]*(\.\d+)?", " _money_ ", t)
    t = re.sub(r"\d[\d,]*(\.\d+)?\s?(£|\$|₹|rs|inr|pounds|rupees)", " _money_ ", t)
    t = re.sub(r"\+?\d[\d\s-]{8,}\d", " _phone_ ", t)
    t = re.sub(r"\b\d{4,6}\b", " _code_ ", t)
    t = re.sub(r"\d+", " _num_ ", t)
    return t
