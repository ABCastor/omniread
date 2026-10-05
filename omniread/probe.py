"""Tier-0 companion probes for structured representations advertised by a page."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

from selectolax.parser import HTMLParser

from ._html import _attr, extract_json_ld
from .types import JsonValue

_FEED_TYPES = {
    "application/atom+xml",
    "application/feed+json",
    "application/jsonfeed+json",
    "application/rss+xml",
}
_OEMBED_TYPES = {"application/json+oembed", "text/xml+oembed"}


@dataclass(frozen=True, slots=True)
class StructuredDataProbe:
    """Structured facts found without claiming they contain the full page."""

    json_ld: JsonValue
    feed_urls: tuple[str, ...]
    oembed_urls: tuple[str, ...]

    def as_json(self) -> JsonValue:
        """Return the probe in the public JSON-compatible result shape."""

        if self.json_ld is None and not self.feed_urls and not self.oembed_urls:
            return None
        return {
            "json_ld": self.json_ld,
            "feeds": list(self.feed_urls),
            "oembed": list(self.oembed_urls),
        }


def probe_structured_data(raw_html: str, *, base_url: str) -> StructuredDataProbe:
    """Collect JSON-LD and advertised feed/oEmbed endpoints as companion evidence.

    This function deliberately does not fetch an advertised endpoint or decide
    completeness. A source recipe may use one of these representations only when
    it also owns an independent manifest capable of verifying that representation.
    """

    tree = HTMLParser(raw_html)
    feeds: list[str] = []
    oembed: list[str] = []
    for node in tree.css("link[href]"):
        media_type = _attr(node, "type").split(";", 1)[0].strip().lower()
        href = _attr(node, "href").strip()
        if not href:
            continue
        resolved = urljoin(base_url, href)
        if media_type in _FEED_TYPES:
            feeds.append(resolved)
        elif media_type in _OEMBED_TYPES:
            oembed.append(resolved)
    return StructuredDataProbe(
        json_ld=extract_json_ld(raw_html),
        feed_urls=tuple(dict.fromkeys(feeds)),
        oembed_urls=tuple(dict.fromkeys(oembed)),
    )
