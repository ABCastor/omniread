from __future__ import annotations

from omniread import tokens
from omniread.tokens import (
    build_outline,
    render_outline_head,
    truncate_to_budget,
)

MARKDOWN = """# Overview

Opening material that belongs to the first whole section.

## Data

| Name | Value |
|---|---|
| alpha | 1 |

## Example

```python
# Not a heading
print("hello")
```

## Example

Closing material.
"""


def test_outline_builds_ordered_heading_tree_and_content_flags() -> None:
    outline = build_outline(MARKDOWN)

    assert [(section.heading, section.level) for section in outline.sections] == [
        ("Overview", 1),
        ("Data", 2),
        ("Example", 2),
        ("Example", 2),
    ]
    assert [section.anchor for section in outline.sections] == [
        "overview",
        "data",
        "example",
        "example-2",
    ]
    assert all(section.token_count > 0 for section in outline.sections)
    assert all(section.char_count > 0 for section in outline.sections)
    assert outline.total_token_count > 0
    assert outline.has_tables is True
    assert outline.has_code is True
    assert outline.has_pagination is False


def test_outline_first_head_names_counts_anchors_and_flags() -> None:
    head = render_outline_head(build_outline(MARKDOWN))

    assert head.startswith("# Outline")
    assert "Approximate tokens:" in head
    assert "tables, code" in head
    assert "[#overview]" in head


def test_budget_truncation_stops_at_first_non_fitting_section() -> None:
    outline = build_outline(MARKDOWN)
    first_section_budget = outline.sections[0].token_count

    bounded = truncate_to_budget(MARKDOWN, first_section_budget)

    assert bounded.truncated is True
    assert bounded.content.startswith("# Overview")
    assert "## Data" not in bounded.content
    assert bounded.omitted == ["data", "example", "example-2"]


def test_budget_never_cuts_an_oversized_first_section_midway() -> None:
    outline = build_outline(MARKDOWN)

    bounded = truncate_to_budget(MARKDOWN, outline.sections[0].token_count - 1)

    assert bounded.content == ""
    assert bounded.truncated is True
    assert bounded.omitted == ["overview", "data", "example", "example-2"]


def test_content_within_budget_is_unchanged() -> None:
    outline = build_outline(MARKDOWN)

    bounded = truncate_to_budget(MARKDOWN, outline.total_token_count)

    assert bounded.content == MARKDOWN
    assert bounded.truncated is False
    assert bounded.omitted == []


def test_token_counter_does_not_download_when_encoding_table_is_absent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        tokens.tiktoken,
        "get_encoding",
        lambda _: (_ for _ in ()).throw(AssertionError("network-backed loader called")),
    )
    tokens._cached_encoding.cache_clear()

    assert tokens.count_tokens("Offline token accounting remains approximate.") > 0
    assert tokens.tokenizer_backend() == "utf8-bytes/4 fallback"

    tokens._cached_encoding.cache_clear()

