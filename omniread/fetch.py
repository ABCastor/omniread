"""Tier-1 static retrieval through curl_cffi browser impersonation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urljoin, urlsplit

from curl_cffi import requests
from selectolax.parser import HTMLParser

from ._html import _attr, _is_hidden, declared_word_count, extract_json_ld, visible_dom_snapshot
from .types import PolicyError, RetrievalError


@dataclass(frozen=True, slots=True)
class BlockSignal:
    """A non-raising signal that the fetched representation is a block page."""

    detected: bool
    reasons: tuple[str, ...] = ()
    auth_required: bool = False

    def __bool__(self) -> bool:
        return self.detected


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """Raw Tier-1 bytes decoded to HTML plus response metadata."""

    raw_html: str
    http_status: int
    final_url: str
    headers: dict[str, str]
    block_signal: BlockSignal
    source_urls: tuple[str, ...] = ()


_BLOCK_TITLES = (
    "just a moment",
    "attention required",
    "access denied",
    "security check",
    "verify you are human",
    "robot or human",
    "unusual traffic",
)
_STRUCTURAL_CHALLENGE_MARKERS = (
    'id="challenge-form"',
    "id='challenge-form'",
    "cf-chl-",
    "/cdn-cgi/challenge-platform",
    "cf-browser-verification",
    "g-recaptcha",
    "hcaptcha",
    "data-sitekey=",
)
_VISIBLE_CHALLENGE_MARKERS = (
    "checking your browser",
    "verify you are human",
    "complete the security check",
    "enable javascript and cookies to continue",
    "confirm you are not a robot",
)
_AUTHWALL_VISIBLE_MARKERS = (
    "subscribe to continue reading",
    "subscribe to unlock this article",
    "sign in to continue reading",
    "log in to continue reading",
    "register to continue reading",
    "this article is for subscribers",
    "this content is only available to subscribers",
    "already a subscriber? sign in",
    "you have reached your article limit",
    "unlock this article",
    "subscription required",
)
_AUTHWALL_PATH_MARKERS = ("/authwall", "/login", "/signin", "/sign-in")
_BLOCK_HTTP_STATUSES = {401, 402, 403, 407, 429, 999}
_MAX_CLIENT_REDIRECT_HOPS = 2
_MAX_REDIRECT_SHELL_BYTES = 16_384
_MAX_REDIRECT_SHELL_WORDS = 40


def fetch_static(url: str, *, timeout: float = 30.0) -> FetchedPage:
    """Fetch with browser TLS impersonation and bounded thin-shell redirects."""

    _validate_http_url(url)
    next_url = url
    source_urls: list[str] = [url]
    response = None
    for hop in range(_MAX_CLIENT_REDIRECT_HOPS + 1):
        try:
            response = requests.get(
                next_url,
                impersonate="chrome",
                timeout=timeout,
                allow_redirects=True,
            )
        except requests.errors.RequestsError as exc:
            raise RetrievalError(f"Tier-1 fetch failed for {next_url}: {exc}") from exc

        final_url = str(response.url)
        if final_url not in source_urls:
            source_urls.append(final_url)
        raw_html = response.text or ""
        status = int(response.status_code)
        target = (
            _thin_client_redirect_target(raw_html, base_url=final_url)
            if 200 <= status < 300
            else None
        )
        if target is None or hop == _MAX_CLIENT_REDIRECT_HOPS or target in source_urls:
            break
        source_urls.append(target)
        next_url = target

    if response is None:  # pragma: no cover - the bounded loop always executes
        raise RetrievalError(f"Tier-1 fetch ended without a response for {url}")
    raw_html = response.text or ""
    status = int(response.status_code)
    return FetchedPage(
        raw_html=raw_html,
        http_status=status,
        final_url=str(response.url),
        headers={str(key): str(value) for key, value in response.headers.items()},
        block_signal=detect_block_page(
            raw_html,
            http_status=status,
            final_url=str(response.url),
        ),
        source_urls=tuple(source_urls),
    )


def _validate_http_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise PolicyError("OmniRead only fetches absolute HTTP(S) URLs")


def _thin_client_redirect_target(raw_html: str, *, base_url: str) -> str | None:
    if len(raw_html.encode("utf-8")) > _MAX_REDIRECT_SHELL_BYTES:
        return None
    snapshot = visible_dom_snapshot(raw_html)
    if snapshot.has_content_container or snapshot.word_count > _MAX_REDIRECT_SHELL_WORDS:
        return None

    tree = HTMLParser(raw_html)
    targets: list[str] = []
    for node in tree.css("meta[http-equiv][content]"):
        if _attr(node, "http-equiv").strip().lower() != "refresh":
            continue
        match = re.search(
            r"(?:^|;)\s*url\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^;]+))\s*$",
            _attr(node, "content"),
            re.I,
        )
        if match:
            targets.append(next(value for value in match.groups() if value is not None).strip())

    location_re = re.compile(
        r"(?:window\.)?location(?:\.replace\(\s*|\.href\s*=\s*)([\"'])(.*?)\1",
        re.I | re.S,
    )
    for node in tree.css("script"):
        targets.extend(match.group(2).strip() for match in location_re.finditer(node.text()))

    resolved: list[str] = []
    for target in targets:
        candidate = urljoin(base_url, target)
        parts = urlsplit(candidate)
        if parts.scheme in {"http", "https"} and parts.netloc:
            resolved.append(candidate)
    unique = tuple(dict.fromkeys(resolved))
    return unique[0] if len(unique) == 1 else None


def detect_block_page(
    raw_html: str,
    *,
    http_status: int,
    final_url: str | None = None,
) -> BlockSignal:
    """Detect challenge/consent representations, including deceptive HTTP 200s.

    Generic consent-script snippets are deliberately not enough. A consent signal
    requires a short, wall-like visible document so ordinary cookie banners do not
    poison otherwise complete articles.
    """

    snapshot = visible_dom_snapshot(raw_html)
    raw_lower = raw_html.lower()
    visible_lower = snapshot.text.lower()
    title_lower = snapshot.title.lower().strip(" .!\t\r\n")
    reasons: list[str] = []
    auth_reasons: list[str] = []

    if http_status in _BLOCK_HTTP_STATUSES:
        reasons.append(f"HTTP {http_status} is a blocked or access-limited response")

    if any(title_lower.startswith(title) for title in _BLOCK_TITLES):
        reasons.append(f"known block title: {snapshot.title or '(empty title)'}")

    structural = any(marker in raw_lower for marker in _STRUCTURAL_CHALLENGE_MARKERS)
    visible_challenge = any(marker in visible_lower for marker in _VISIBLE_CHALLENGE_MARKERS)
    if structural and (snapshot.word_count < 250 or visible_challenge):
        reasons.append("challenge markup dominates the response")
    elif visible_challenge and snapshot.word_count < 250:
        reasons.append("challenge language dominates the visible response")

    consent_language = (
        "consent" in visible_lower
        and any(
            phrase in visible_lower
            for phrase in ("accept all", "manage preferences", "privacy choices", "reject all")
        )
    )
    if consent_language and snapshot.word_count < 180 and not snapshot.has_content_container:
        reasons.append("short consent wall without an article/main container")

    tree = HTMLParser(raw_html)
    auth_nodes = []
    for selector in (
        '[class*="paywall"]',
        '[id*="paywall"]',
        '[class*="subscription-wall"]',
        '[id*="subscription-wall"]',
        '[class*="login-wall"]',
        '[id*="login-wall"]',
        '[class*="regwall"]',
        '[id*="regwall"]',
        '[class*="content-gate"]',
        '[id*="content-gate"]',
        '[class*="subscriber-only"]',
        '[id*="subscriber-only"]',
        '[class*="gateway-content"]',
        '[id*="gateway-content"]',
    ):
        auth_nodes.extend(node for node in tree.css(selector) if not _is_hidden(node))
    structural_auth_text = " ".join(node.text(separator=" ") for node in auth_nodes).lower()
    auth_controls = [
        node
        for node in tree.css('input[type="password"], form[action]')
        if not _is_hidden(node)
        and (
            _attr(node, "type").lower() == "password"
            or any(
                marker in _attr(node, "action").lower()
                for marker in _AUTHWALL_PATH_MARKERS
            )
        )
    ]
    visible_auth = any(marker in visible_lower for marker in _AUTHWALL_VISIBLE_MARKERS)
    structural_auth = bool(auth_nodes) and (
        snapshot.word_count < 600
        or any(marker in structural_auth_text for marker in _AUTHWALL_VISIBLE_MARKERS)
    )
    final_path = urlsplit(final_url).path.lower() if final_url else ""
    auth_redirect = any(marker in final_path for marker in _AUTHWALL_PATH_MARKERS)
    thin_authwall_stub = (
        snapshot.word_count < 60
        and not snapshot.has_content_container
        and any(
            marker in raw_lower
            for marker in (
                "/authwall",
                "subscription-wall",
                "paywall",
                "regwall",
                "gateway-content",
            )
        )
    )
    if (
        visible_auth
        and snapshot.word_count < 600
        and (bool(auth_nodes) or bool(auth_controls))
    ):
        auth_reasons.append("subscription or sign-in language dominates the response")
    if structural_auth:
        auth_reasons.append("a paywall or login-wall container dominates the response")
    if auth_redirect:
        auth_reasons.append(f"the final URL is an authentication route: {final_path}")
    if thin_authwall_stub:
        auth_reasons.append("a thin redirect stub points to an authentication wall")
    reasons.extend(f"authentication wall: {reason}" for reason in auth_reasons)

    structured_data = extract_json_ld(raw_html)
    declared = declared_word_count(raw_html, structured_data)
    if declared is not None and declared >= 250:
        extreme_floor = max(40, int(declared * 0.15))
        if snapshot.word_count < extreme_floor:
            reasons.append(
                f"declared article has {declared} words but only "
                f"{snapshot.word_count} visible words arrived"
            )

    if reasons and http_status == 200:
        reasons.insert(0, "HTTP 200 response contains block-page evidence")

    return BlockSignal(
        detected=bool(reasons),
        reasons=tuple(dict.fromkeys(reasons)),
        auth_required=bool(auth_reasons),
    )
