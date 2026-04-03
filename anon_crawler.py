"""
anon_crawler.py — Privacy-focused anonymous web crawling for libaix.

Provides anonymous crawling capabilities with:
  • Rotating user-agent strings (large pool of realistic browser UAs)
  • Random request delays with jitter to avoid fingerprinting
  • Referrer spoofing / removal
  • Cookie isolation per crawl session
  • DNS-over-HTTPS support (configurable)
  • Request header randomization
  • Proxy support (HTTP/SOCKS when available)
  • Accept-Language rotation
  • Cache-control randomization
  • No persistent cookies or tracking tokens
  • Session isolation (each crawl is independent)
  • Clean metadata stripping from responses
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import random
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from file_processor import classify_domain, generate_qa_from_text
from site_crawler import _TextExtractor, EXTRA_KNOWLEDGE_DIR

# ── Paths ─────────────────────────────────────────────────────────────

ANON_CONFIG_PATH = Path("data/anon_config.json")
ANON_STATS_PATH = Path("data/anon_stats.json")
MAX_PAGE_SIZE = 2 * 1024 * 1024  # 2 MB

# ── SSRF protection ──────────────────────────────────────────────────

_SSRF_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "[::1]", "0.0.0.0",
    "169.254.169.254",          # AWS/GCP/Azure metadata
    "metadata.google.internal",
    "metadata.internal",
}
_SSRF_BLOCKED_PREFIXES = (
    "http://10.", "https://10.",
    "http://172.16.", "https://172.16.", "http://172.17.", "https://172.17.",
    "http://172.18.", "https://172.18.", "http://172.19.", "https://172.19.",
    "http://172.20.", "https://172.20.", "http://172.21.", "https://172.21.",
    "http://172.22.", "https://172.22.", "http://172.23.", "https://172.23.",
    "http://172.24.", "https://172.24.", "http://172.25.", "https://172.25.",
    "http://172.26.", "https://172.26.", "http://172.27.", "https://172.27.",
    "http://172.28.", "https://172.28.", "http://172.29.", "https://172.29.",
    "http://172.30.", "https://172.30.", "http://172.31.", "https://172.31.",
    "http://192.168.", "https://192.168.",
    "file://", "ftp://", "gopher://",
)


def _is_safe_url(url: str) -> bool:
    """Check if a URL is safe to crawl (not targeting internal networks)."""
    url_lower = url.lower()
    try:
        after_scheme = url_lower.split("://", 1)[1] if "://" in url_lower else url_lower
        host = after_scheme.split("/")[0].split(":")[0].split("?")[0]
    except (IndexError, ValueError):
        return False
    if host in _SSRF_BLOCKED_HOSTS:
        return False
    if url_lower.startswith(_SSRF_BLOCKED_PREFIXES):
        return False
    # Reject numeric-only hosts that resolve to private ranges
    if re.match(r"^\d+$", host):
        return False
    return True


# ── User-Agent pool ──────────────────────────────────────────────────

USER_AGENT_POOL: list[str] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    # Firefox on Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    # Edge on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

ACCEPT_LANGUAGE_POOL: list[str] = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,fr;q=0.8",
    "en-US,en;q=0.9,de;q=0.8",
    "en-US,en;q=0.9,es;q=0.8",
    "en-US,en;q=0.9,ja;q=0.8",
    "en-US,en;q=0.9,pt-BR;q=0.8",
    "en-US,en;q=0.9,zh-CN;q=0.8",
    "en-US,en;q=0.9,ko;q=0.8",
    "en-US,en;q=0.9,it;q=0.8",
    "en-US,en;q=0.9,nl;q=0.8",
    "en-US,en;q=0.9,ru;q=0.8",
]

_ACCEPT_POOL = [
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
]

_ACCEPT_ENCODING_POOL = [
    "gzip, deflate, br",
    "gzip, deflate",
    "gzip, deflate, br, zstd",
]

_CACHE_CONTROL_POOL = [
    "no-cache",
    "max-age=0",
    "no-store",
    "",  # omit header entirely
]

_REFERRER_POLICIES = {"none", "same-origin", "spoofed"}

_SPOOFED_REFERRERS = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://search.yahoo.com/",
    "https://www.ecosia.org/",
]

# ── Module-level crawl statistics ────────────────────────────────────

_stats: dict = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "bytes_downloaded": 0,
    "pages_crawled": 0,
    "entries_extracted": 0,
    "domains_visited": [],
    "last_request_at": None,
}

# Rate-limit tracker: domain → last_request_timestamp
_domain_rate_limits: dict[str, float] = {}

# DNS cache: hostname → (ip, expiry_timestamp)
_dns_cache: dict[str, tuple[str, float]] = {}
_DNS_CACHE_TTL = 300.0  # 5 minutes


# ── Config management ────────────────────────────────────────────────

def _default_anon_config() -> dict:
    return {
        "delay_min": 1.0,
        "delay_max": 5.0,
        "jitter": 0.5,
        "referrer_policy": "none",
        "respect_robots_txt": True,
        "timeout": 20,
        "max_retries": 2,
        "retry_backoff": 2.0,
        "rate_limit_per_domain": 2.0,
        "strip_cookies": True,
        "strip_tracking_params": True,
        "dns_over_https": False,
        "doh_server": "https://cloudflare-dns.com/dns-query",
        "proxies": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_anon_config() -> dict:
    """Load the anonymous crawler configuration from disk."""
    if ANON_CONFIG_PATH.exists():
        try:
            return json.loads(ANON_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return _default_anon_config()


def save_anon_config(config: dict) -> None:
    """Persist the anonymous crawler configuration to disk."""
    ANON_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANON_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ── Proxy management ────────────────────────────────────────────────

def add_proxy(proxy_url: str) -> bool:
    """Add a proxy URL (http://... or socks5://...) to the rotation pool."""
    parsed = urllib.parse.urlparse(proxy_url)
    if parsed.scheme not in ("http", "https", "socks5", "socks4"):
        return False
    if not parsed.hostname:
        return False
    config = load_anon_config()
    proxies = config.get("proxies", [])
    # Avoid duplicates
    for p in proxies:
        if p.get("url") == proxy_url:
            return False
    proxies.append({
        "url": proxy_url,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "fail_count": 0,
        "success_count": 0,
        "enabled": True,
    })
    config["proxies"] = proxies
    save_anon_config(config)
    return True


def remove_proxy(proxy_url: str) -> bool:
    """Remove a proxy from the pool by URL."""
    config = load_anon_config()
    proxies = config.get("proxies", [])
    original_len = len(proxies)
    config["proxies"] = [p for p in proxies if p.get("url") != proxy_url]
    if len(config["proxies"]) < original_len:
        save_anon_config(config)
        return True
    return False


def list_proxies() -> list[dict]:
    """Return the current list of configured proxies."""
    config = load_anon_config()
    return config.get("proxies", [])


# ── DNS helpers ──────────────────────────────────────────────────────

def _resolve_dns(hostname: str, config: dict) -> str | None:
    """Resolve hostname, using cache and optionally DNS-over-HTTPS."""
    now = time.time()
    cached = _dns_cache.get(hostname)
    if cached and cached[1] > now:
        return cached[0]

    ip: str | None = None
    if config.get("dns_over_https"):
        ip = _doh_resolve(hostname, config.get("doh_server", ""))
    if not ip:
        try:
            ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            return None
    if ip:
        _dns_cache[hostname] = (ip, now + _DNS_CACHE_TTL)
    return ip


def _doh_resolve(hostname: str, doh_server: str) -> str | None:
    """Resolve hostname via DNS-over-HTTPS (JSON API)."""
    if not doh_server:
        return None
    try:
        params = urllib.parse.urlencode({"name": hostname, "type": "A"})
        url = f"{doh_server}?{params}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/dns-json",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        for answer in data.get("Answer", []):
            if answer.get("type") == 1:  # A record
                return answer.get("data")
    except Exception:
        pass
    return None


# ── Robots.txt support ───────────────────────────────────────────────

_robots_cache: dict[str, tuple[list[str], float]] = {}
_ROBOTS_CACHE_TTL = 600.0  # 10 minutes


def _is_allowed_by_robots(url: str, config: dict) -> bool:
    """Check robots.txt for the given URL. Returns True if allowed."""
    if not config.get("respect_robots_txt", True):
        return True
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = f"{base}/robots.txt"
    now = time.time()

    cached = _robots_cache.get(base)
    if cached and cached[1] > now:
        disallowed = cached[0]
    else:
        disallowed = _fetch_robots_disallowed(robots_url)
        _robots_cache[base] = (disallowed, now + _ROBOTS_CACHE_TTL)

    path = parsed.path or "/"
    for rule in disallowed:
        if path.startswith(rule):
            return False
    return True


def _fetch_robots_disallowed(robots_url: str) -> list[str]:
    """Fetch robots.txt and return Disallow paths for * user-agent."""
    disallowed: list[str] = []
    try:
        req = urllib.request.Request(robots_url, headers={
            "User-Agent": random.choice(USER_AGENT_POOL),
        })
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            text = resp.read(64 * 1024).decode("utf-8", errors="replace")
        in_wildcard = False
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                in_wildcard = agent == "*"
            elif in_wildcard and line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    disallowed.append(path)
    except Exception:
        pass  # If we can't fetch robots.txt, assume allowed
    return disallowed


# ── URL cleaning ─────────────────────────────────────────────────────

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "twclid",
    "mc_cid", "mc_eid", "yclid", "ref", "_ga", "_gl",
}


def _strip_tracking_params(url: str) -> str:
    """Remove known tracking query parameters from a URL."""
    parsed = urllib.parse.urlparse(url)
    if not parsed.query:
        return url
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    cleaned = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
    new_query = urllib.parse.urlencode(cleaned, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


# ── Header randomization ────────────────────────────────────────────

def _build_random_headers(config: dict, target_url: str) -> dict[str, str]:
    """Build a randomized set of HTTP headers for a request."""
    headers: dict[str, str] = {
        "User-Agent": random.choice(USER_AGENT_POOL),
        "Accept": random.choice(_ACCEPT_POOL),
        "Accept-Language": random.choice(ACCEPT_LANGUAGE_POOL),
        "Accept-Encoding": random.choice(_ACCEPT_ENCODING_POOL),
    }

    # Cache-control
    cc = random.choice(_CACHE_CONTROL_POOL)
    if cc:
        headers["Cache-Control"] = cc

    # Connection header
    headers["Connection"] = random.choice(["keep-alive", "close"])

    # Upgrade-Insecure-Requests (browsers usually send this)
    if random.random() > 0.3:
        headers["Upgrade-Insecure-Requests"] = "1"

    # DNT (Do Not Track)
    if random.random() > 0.5:
        headers["DNT"] = "1"

    # Sec-Fetch headers (modern browser fingerprint)
    if random.random() > 0.4:
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = random.choice(["none", "cross-site"])
        headers["Sec-Fetch-User"] = "?1"

    # Referrer policy
    policy = config.get("referrer_policy", "none")
    if policy == "spoofed":
        headers["Referer"] = random.choice(_SPOOFED_REFERRERS)
    elif policy == "same-origin":
        parsed = urllib.parse.urlparse(target_url)
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    # "none" → omit Referer entirely

    return headers


# ── Rate limiting ────────────────────────────────────────────────────

def _wait_for_rate_limit(domain: str, config: dict) -> None:
    """Enforce per-domain rate limiting with random jitter."""
    min_interval = config.get("rate_limit_per_domain", 2.0)
    last = _domain_rate_limits.get(domain, 0.0)
    elapsed = time.time() - last
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    # Add random delay with jitter
    delay_min = config.get("delay_min", 1.0)
    delay_max = config.get("delay_max", 5.0)
    jitter = config.get("jitter", 0.5)
    delay = random.uniform(delay_min, delay_max) + random.uniform(-jitter, jitter)
    delay = max(0.1, delay)
    time.sleep(delay)
    _domain_rate_limits[domain] = time.time()


# ── Core fetch ───────────────────────────────────────────────────────

def anon_fetch(url: str, config: dict | None = None) -> dict:
    """
    Fetch a URL anonymously with privacy protections.

    Returns a dict with keys:
        status  — HTTP status code (0 on network error)
        text    — response body as text
        headers — response headers as dict
        url     — final URL (after redirects / cleaning)
        error   — error message if any (empty string on success)
    """
    if config is None:
        config = load_anon_config()

    # SSRF protection
    if not _is_safe_url(url):
        return {"status": 0, "text": "", "headers": {}, "url": url,
                "error": "URL targets an internal or reserved network address"}

    # Strip tracking params
    if config.get("strip_tracking_params", True):
        url = _strip_tracking_params(url)

    # Robots.txt check
    if not _is_allowed_by_robots(url, config):
        return {"status": 0, "text": "", "headers": {}, "url": url,
                "error": "Blocked by robots.txt"}

    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc

    # DNS pre-resolution (populate cache)
    hostname = parsed.hostname or ""
    if hostname:
        _resolve_dns(hostname, config)

    # Rate limit
    _wait_for_rate_limit(domain, config)

    # Cookie isolation — fresh jar per request
    cookie_jar = http.cookiejar.CookieJar()
    cookie_proc = urllib.request.HTTPCookieProcessor(cookie_jar)

    # Proxy setup
    proxy_entry = _select_proxy(config)
    handlers: list = [cookie_proc]
    if proxy_entry:
        proxy_url = proxy_entry["url"]
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy_url,
            "https": proxy_url,
        })
        handlers.append(proxy_handler)

    opener = urllib.request.build_opener(*handlers)
    headers = _build_random_headers(config, url)
    timeout = config.get("timeout", 20)
    max_retries = config.get("max_retries", 2)
    backoff = config.get("retry_backoff", 2.0)

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = opener.open(req, timeout=timeout)
            content_type = resp.headers.get("Content-Type", "")
            if "text/" not in content_type and "application/json" not in content_type:
                return {"status": resp.status, "text": "", "headers": {},
                        "url": url, "error": f"Non-text content type: {content_type}"}
            body = resp.read(MAX_PAGE_SIZE).decode("utf-8", errors="replace")

            # Sanitize response headers — strip Set-Cookie, tracking headers
            resp_headers = dict(resp.headers.items())
            if config.get("strip_cookies", True):
                resp_headers.pop("Set-Cookie", None)
                resp_headers.pop("set-cookie", None)

            # Update stats
            _stats["total_requests"] += 1
            _stats["successful_requests"] += 1
            _stats["bytes_downloaded"] += len(body)
            _stats["last_request_at"] = datetime.now(timezone.utc).isoformat()
            if domain not in _stats["domains_visited"]:
                _stats["domains_visited"].append(domain)
            if proxy_entry:
                proxy_entry["success_count"] = proxy_entry.get("success_count", 0) + 1

            return {
                "status": resp.status,
                "text": body,
                "headers": resp_headers,
                "url": resp.url or url,
                "error": "",
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            _stats["total_requests"] += 1
            _stats["failed_requests"] += 1
            # Don't retry on 4xx client errors (except 429)
            if 400 <= exc.code < 500 and exc.code != 429:
                break
        except Exception as exc:
            last_error = str(exc)
            _stats["total_requests"] += 1
            _stats["failed_requests"] += 1
            if proxy_entry:
                proxy_entry["fail_count"] = proxy_entry.get("fail_count", 0) + 1

        if attempt < max_retries:
            time.sleep(backoff * (attempt + 1) + random.uniform(0, 1))

    return {"status": 0, "text": "", "headers": {}, "url": url, "error": last_error}


def _select_proxy(config: dict) -> dict | None:
    """Select a random enabled proxy from the pool, or None."""
    proxies = [p for p in config.get("proxies", []) if p.get("enabled", True)]
    if not proxies:
        return None
    return random.choice(proxies)


# ── HTML helpers ─────────────────────────────────────────────────────

def _extract_text(html: str) -> str:
    """Extract plain text from HTML using site_crawler's _TextExtractor."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.get_text()


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract absolute links from HTML."""
    links: list[str] = []
    for match in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', html, re.IGNORECASE):
        href = match.group(1)
        try:
            absolute = urllib.parse.urljoin(base_url, href)
            absolute = absolute.split("#")[0]
            if absolute.startswith(("http://", "https://")):
                links.append(absolute)
        except Exception:
            continue
    return links


def _is_same_domain(url: str, base_domain: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc.endswith(f".{base_domain}")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text.strip()
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    if last_period > max_len // 3:
        return truncated[: last_period + 1].strip()
    return truncated.strip() + "."


# ── Single-page crawl ───────────────────────────────────────────────

def anon_crawl_page(url: str, topic: str, config: dict | None = None) -> dict:
    """
    Anonymously fetch a single page and extract topic-relevant Q&A.

    Returns dict with status, entries, and page metadata.
    """
    if config is None:
        config = load_anon_config()

    result = anon_fetch(url, config)
    if result["error"]:
        return {
            "status": "error",
            "error": result["error"],
            "entries": 0,
            "entries_data": [],
        }

    html = result["text"]
    text = _extract_text(html)

    if len(text) < 50:
        return {
            "status": "no_content",
            "error": "Page contained insufficient text",
            "entries": 0,
            "entries_data": [],
        }

    # Generate Q&A entries
    entries = generate_qa_from_text(text)
    seen: set[str] = set()
    unique_entries: list[dict] = []
    for entry in entries:
        ql = entry["question"].lower()
        if ql not in seen:
            seen.add(ql)
            entry["source"] = f"anon:{urllib.parse.urlparse(url).netloc}"
            unique_entries.append(entry)

    # Summary entry
    if len(text) > 100:
        domain = classify_domain(text)
        summary_q = f"What does this page describe about {topic}?"
        if summary_q.lower() not in seen:
            unique_entries.append({
                "question": summary_q,
                "answer": _truncate(text, 500),
                "domain": domain,
                "source": f"anon:{urllib.parse.urlparse(url).netloc}",
            })

    _stats["pages_crawled"] += 1
    _stats["entries_extracted"] += len(unique_entries)

    return {
        "status": "success" if unique_entries else "no_results",
        "entries": len(unique_entries),
        "entries_data": unique_entries,
        "http_status": result["status"],
        "text_length": len(text),
    }


# ── Multi-page site crawl ───────────────────────────────────────────

def anon_crawl_site(
    url: str,
    topic: str,
    max_pages: int = 20,
    max_depth: int = 2,
    config: dict | None = None,
) -> dict:
    """
    Anonymously crawl a website, following internal links up to max_depth.

    Returns dict with aggregated Q&A entries and crawl statistics.
    """
    if config is None:
        config = load_anon_config()

    parsed_start = urllib.parse.urlparse(url)
    base_domain = parsed_start.netloc
    if not base_domain:
        return {"status": "error", "error": "Invalid URL", "entries": 0,
                "entries_data": []}

    visited: set[str] = set()
    all_entries: list[dict] = []
    seen_questions: set[str] = set()
    pages_crawled = 0
    pages_relevant = 0
    total_bytes = 0
    errors: list[str] = []

    # BFS queue: (url, depth)
    queue: list[tuple[str, int]] = [(url, 0)]

    while queue and pages_crawled < max_pages:
        current_url, depth = queue.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        # SSRF check each URL
        if not _is_safe_url(current_url):
            continue

        result = anon_fetch(current_url, config)
        if result["error"]:
            errors.append(f"{current_url}: {result['error']}")
            continue

        pages_crawled += 1
        html = result["text"]
        text = _extract_text(html)
        total_bytes += len(text)

        # Check relevance
        topic_words = topic.lower().split()
        text_lower = text.lower()
        is_relevant = sum(1 for w in topic_words if w in text_lower) >= max(
            1, len(topic_words) // 2
        )

        if is_relevant and len(text) > 50:
            pages_relevant += 1
            entries = generate_qa_from_text(text)
            for entry in entries:
                ql = entry["question"].lower()
                if ql not in seen_questions:
                    seen_questions.add(ql)
                    entry["source"] = f"anon:{base_domain}:{current_url}"
                    all_entries.append(entry)

        # Follow internal links within depth limit
        if depth < max_depth:
            links = _extract_links(html, current_url)
            for link in links:
                if link not in visited and _is_same_domain(link, base_domain):
                    queue.append((link, depth + 1))

        # Discard raw data
        del html, text

    # Save knowledge if we got results
    saved_path = None
    if all_entries:
        saved_path = _save_anon_knowledge(all_entries, base_domain, topic)

    _stats["pages_crawled"] += pages_crawled
    _stats["entries_extracted"] += len(all_entries)

    result_dict: dict = {
        "status": "success" if all_entries else "no_results",
        "entries": len(all_entries),
        "entries_data": all_entries,
        "file": str(saved_path) if saved_path else None,
        "samples": all_entries[:3] if all_entries else [],
        "stats": {
            "pages_crawled": pages_crawled,
            "pages_relevant": pages_relevant,
            "total_text_bytes": total_bytes,
            "unique_entries": len(all_entries),
            "errors": len(errors),
        },
    }
    if errors:
        result_dict["errors"] = errors[:10]  # cap error list
    return result_dict


# ── Persistence ──────────────────────────────────────────────────────

def _save_anon_knowledge(entries: list[dict], domain: str, topic: str) -> Path:
    """Save extracted knowledge to the extra_knowledge directory."""
    EXTRA_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    safe_domain = re.sub(r"[^\w\-]", "_", domain)
    safe_topic = re.sub(r"[^\w\-]", "_", topic.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fp = EXTRA_KNOWLEDGE_DIR / f"anon_{safe_domain}_{safe_topic}_{ts}.json"
    fp.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return fp


# ── Statistics ───────────────────────────────────────────────────────

def get_anon_stats() -> dict:
    """Return current crawl statistics for the anonymous crawler."""
    return {
        "total_requests": _stats["total_requests"],
        "successful_requests": _stats["successful_requests"],
        "failed_requests": _stats["failed_requests"],
        "bytes_downloaded": _stats["bytes_downloaded"],
        "pages_crawled": _stats["pages_crawled"],
        "entries_extracted": _stats["entries_extracted"],
        "domains_visited": list(_stats["domains_visited"]),
        "last_request_at": _stats["last_request_at"],
        "dns_cache_size": len(_dns_cache),
        "robots_cache_size": len(_robots_cache),
        "proxies_configured": len(load_anon_config().get("proxies", [])),
    }


# ── Anonymity test ───────────────────────────────────────────────────

def test_anonymity() -> dict:
    """
    Run a basic anonymity check by inspecting what headers a public
    echo service would see.  Uses httpbin.org/headers (public, free).

    Returns a dict describing the anonymity posture.
    """
    config = load_anon_config()
    result = anon_fetch("https://httpbin.org/headers", config)

    report: dict = {
        "status": "unknown",
        "checks": {},
        "warnings": [],
    }

    if result["error"]:
        report["status"] = "error"
        report["error"] = result["error"]
        return report

    try:
        data = json.loads(result["text"])
        seen_headers = data.get("headers", {})
    except (json.JSONDecodeError, KeyError):
        report["status"] = "error"
        report["error"] = "Could not parse echo response"
        return report

    # Check: user-agent is randomized (not default Python)
    ua = seen_headers.get("User-Agent", "")
    report["checks"]["user_agent_randomized"] = "python" not in ua.lower()
    if not report["checks"]["user_agent_randomized"]:
        report["warnings"].append("User-Agent reveals Python runtime")

    # Check: no referrer leaked
    has_referer = "Referer" in seen_headers
    policy = config.get("referrer_policy", "none")
    if policy == "none":
        report["checks"]["referer_clean"] = not has_referer
        if has_referer:
            report["warnings"].append("Referer header is leaking despite 'none' policy")
    else:
        report["checks"]["referer_clean"] = True

    # Check: Accept-Language present (looks like a real browser)
    report["checks"]["accept_language_set"] = bool(seen_headers.get("Accept-Language"))

    # Check: proxy active
    proxies = config.get("proxies", [])
    active_proxies = [p for p in proxies if p.get("enabled")]
    report["checks"]["proxy_active"] = len(active_proxies) > 0
    if not active_proxies:
        report["warnings"].append("No proxy configured — IP is exposed")

    # Overall score
    passed = sum(1 for v in report["checks"].values() if v)
    total = len(report["checks"])
    report["score"] = f"{passed}/{total}"
    report["status"] = "good" if passed >= 3 else "moderate" if passed >= 2 else "poor"

    return report
