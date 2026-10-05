"""Trafilatura-backed main-content extraction for Tier 1."""

from __future__ import annotations

from dataclasses import dataclass

import trafilatura

from ._html import canonical_url, strip_scripts_and_styles
from .probe import probe_structured_data
from .types import JsonValue


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Trafilatura's Markdown plus structured data kept outside the prose."""

    markdown: str
    structured_data: JsonValue
    canonical_url: str


def extract_page(raw_html: str, *, url: str) -> ExtractedPage:
    """Extract clean Markdown while retaining tables, code, headings, and links.

    JSON-LD is parsed first and returned separately. Script and style nodes are
    removed before Trafilatura sees the document, so they cannot consume length or
    token budgets.
    """

    structured_data = probe_structured_data(raw_html, base_url=url).as_json()
    clean_html = strip_scripts_and_styles(raw_html)
    markdown = trafilatura.extract(
        clean_html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        include_links=True,
        favor_recall=True,
    )
    return ExtractedPage(
        markdown=(markdown or "").strip(),
        structured_data=structured_data,
        canonical_url=canonical_url(raw_html, url),
    )
