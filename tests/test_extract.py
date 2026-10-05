from __future__ import annotations

from pathlib import Path

from omniread._html import (
    canonical_url,
    declared_word_count,
    has_pagination,
    visible_dom_snapshot,
)
from omniread.extract import extract_page

FIXTURES = Path(__file__).parent / "fixtures"


def test_trafilatura_preserves_article_structure_and_jsonld_aside() -> None:
    raw_html = (FIXTURES / "normal_article.html").read_text(encoding="utf-8")
    extracted = extract_page(raw_html, url="https://example.test/redirect")

    assert "# Measuring a Reliable Reader" in extracted.markdown
    assert "## Independent evidence" in extracted.markdown
    assert "| Signal | Role |" in extracted.markdown
    assert 'verdict = "unknown"' in extracted.markdown
    assert "```" in extracted.markdown
    assert "STYLE TEXT MUST NEVER" not in extracted.markdown
    assert "SCRIPT TEXT MUST NEVER" not in extracted.markdown
    assert extracted.structured_data is not None
    assert extracted.canonical_url == "https://example.test/articles/reliable-reader"


def test_valueless_html_attributes_are_treated_as_empty_strings() -> None:
    raw_html = (FIXTURES / "valueless_attributes.html").read_text(encoding="utf-8")
    fallback = "https://example.test/attribute-probe"

    snapshot = visible_dom_snapshot(raw_html)

    assert "This visible text must survive" in snapshot.text
    assert canonical_url(raw_html, fallback) == fallback
    assert has_pagination(raw_html, current_url=fallback) is False
    assert declared_word_count(raw_html, None) is None
