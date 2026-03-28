"""
file_processor.py — Extract text from uploaded files and convert to knowledge entries.

Supports: PDF, TXT, MD, CSV, LOG, CONF and pasted text.
Pipeline: Upload → Extract text → Parse Q&A → Classify domain → Return entries.
Original files are deleted by the caller after extraction (space preservation).
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

# Domain keywords for auto-classification
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "networking": [
        "tcp", "udp", "ip address", "router", "switch", "vlan", "ospf", "bgp",
        "subnet", "ethernet", "arp", "mac address", "layer 2", "layer 3", "osi",
        "routing", "bandwidth", "latency", "mpls", "sd-wan", "stp", "lacp",
        "port number", "icmp", "nat", "gateway", "dhcp",
    ],
    "wifi": [
        "wifi", "wi-fi", "802.11", "wireless", "ssid", "access point", "ofdma",
        "mu-mimo", "beamforming", "channel", "2.4 ghz", "5 ghz", "6 ghz",
        "mesh network", "roaming", "band steering", "dfs",
    ],
    "wifi_security": [
        "wpa2", "wpa3", "802.1x", "radius", "eap", "eap-tls", "peap", "psk",
        "rogue ap", "evil twin", "pmf", "krack", "nac", "wids", "wips",
        "mac filtering", "captive portal",
    ],
    "wifi_policy": [
        "guest network", "byod", "content filtering", "bandwidth throttling",
        "qos", "captive portal", "acceptable use", "mdm", "client isolation",
        "dns filtering", "application control", "wmm",
    ],
    "security": [
        "firewall", "ids", "ips", "siem", "zero trust", "vpn", "encryption",
        "malware", "ransomware", "phishing", "ddos", "penetration testing",
        "certificate", "pki", "ipsec", "ssl", "tls", "xss", "sql injection",
        "endpoint detection", "dlp", "casb",
    ],
    "internet": [
        "http", "https", "dns", "url", "web server", "api", "rest", "cloud",
        "cdn", "ftp", "ssh", "tls", "cookie", "websocket", "smtp", "imap",
    ],
    "intranet": [
        "active directory", "ldap", "sso", "single sign-on", "proxy", "dmz",
        "group policy", "kerberos", "ntlm", "nps", "azure ad", "intune",
        "extranet",
    ],
    "troubleshooting": [
        "troubleshoot", "diagnose", "debug", "rssi", "snr", "site survey",
        "wireshark", "traceroute", "ping", "packet loss", "slow network",
    ],
}


def extract_text_from_file(filepath: Path) -> str:
    """Extract text content from a file based on its extension."""
    suffix = filepath.suffix.lower()

    if suffix in (".txt", ".md", ".log", ".conf", ".cfg", ".ini"):
        return filepath.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".csv":
        return _extract_csv(filepath)
    elif suffix == ".pdf":
        return _extract_pdf(filepath)
    elif suffix in (".html", ".xml"):
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        return _strip_tags(raw)
    else:
        return filepath.read_text(encoding="utf-8", errors="replace")


def extract_text_from_string(text: str) -> str:
    """Clean and normalise pasted text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── Format-specific extractors ────────────────────────────────────────

def _extract_csv(filepath: Path) -> str:
    lines: list[str] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            lines.append(" ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(lines)


def _extract_pdf(filepath: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(
            "pypdf is required for PDF extraction. Install: pip install pypdf"
        )
    reader = PdfReader(filepath)
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)
    return "\n\n".join(parts)


def _strip_tags(html: str) -> str:
    """Remove HTML/XML tags, keep text."""
    return re.sub(r"<[^>]+>", " ", html)


# ── Domain classification ─────────────────────────────────────────────

def classify_domain(text: str) -> str:
    """Classify text into a knowledge domain via keyword scoring."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# ── Q&A generation ────────────────────────────────────────────────────

def split_into_chunks(text: str) -> list[str]:
    """Split text into paragraphs (meaningful chunks)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]


def generate_qa_from_text(
    text: str, domain_hint: str = ""
) -> list[dict[str, str]]:
    """Generate Q&A pairs from free text using heuristic patterns."""
    entries: list[dict[str, str]] = []
    chunks = split_into_chunks(text)

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 600:
                continue

            domain = domain_hint or classify_domain(sentence)

            # Pattern 1 — definition: "X is a/an/the Y"
            m = re.match(
                r"^(?:A |An |The )?(.{3,80}?)\s+"
                r"(?:is|are|refers to|stands for)\s+(.{20,})$",
                sentence,
                re.IGNORECASE,
            )
            if m:
                subject = m.group(1).strip().rstrip(",")
                entries.append(
                    {
                        "question": f"What is {subject}?",
                        "answer": _clean_answer(sentence),
                        "domain": domain,
                    }
                )
                continue

            # Pattern 2 — capability: "X provides/enables/supports Y"
            m = re.match(
                r"^(?:A |An |The )?(.{3,80}?)\s+"
                r"(?:provides|enables|supports|allows|offers|uses)\s+(.{20,})$",
                sentence,
                re.IGNORECASE,
            )
            if m:
                subject = m.group(1).strip().rstrip(",")
                entries.append(
                    {
                        "question": f"What does {subject} provide?",
                        "answer": _clean_answer(sentence),
                        "domain": domain,
                    }
                )
                continue

            # Pattern 3 — informational sentences with key terms
            key_terms = _extract_key_terms(sentence)
            if key_terms and len(sentence) > 60:
                entries.append(
                    {
                        "question": f"Tell me about {key_terms[0]}",
                        "answer": _clean_answer(sentence),
                        "domain": domain,
                    }
                )

    # Deduplicate by question
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for entry in entries:
        q = entry["question"].lower()
        if q not in seen:
            seen.add(q)
            unique.append(entry)
    return unique


def _clean_answer(text: str) -> str:
    text = text.strip()
    if not text.endswith("."):
        text += "."
    return text


def _extract_key_terms(text: str) -> list[str]:
    """Extract capitalised terms and acronyms."""
    terms: list[str] = []
    terms.extend(re.findall(r"\b[A-Z]{2,}[a-z]*\b", text))
    terms.extend(re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text))
    return terms[:3]


# ── High-level pipeline ──────────────────────────────────────────────

def process_file(
    filepath: Path, domain_hint: str = ""
) -> tuple[list[dict[str, str]], str]:
    """Extract text → generate Q&A → return (entries, preview)."""
    text = extract_text_from_file(filepath)
    entries = generate_qa_from_text(text, domain_hint)
    preview = text[:500] + ("…" if len(text) > 500 else "")
    return entries, preview


def process_pasted_text(
    text: str, domain_hint: str = ""
) -> list[dict[str, str]]:
    """Process pasted text into knowledge entries."""
    clean = extract_text_from_string(text)
    return generate_qa_from_text(clean, domain_hint)
