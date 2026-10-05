"""Approximate token accounting, outlines, and honest boundary truncation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import tiktoken
from tiktoken.core import Encoding

from ._html import MD_FENCE_RE, MD_HEADING_RE
from .types import BudgetError, CostToComplete, Outline, Section

_CL100K_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
_CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
_PAGINATION_RE = re.compile(
    r"(?:\bpage\s+\d+\s+of\s+\d+\b|^\s*\[(?:next|older|more)\]\([^\n]+\)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class TruncatedContent:
    """A section-boundary prefix and the anchors deliberately left out."""

    content: str
    truncated: bool
    omitted: list[str]


@dataclass(frozen=True, slots=True)
class _SectionChunk:
    section: Section
    markdown: str


def count_tokens(text: str) -> int:
    """Return an approximate token count without ever fetching tokenizer data.

    A locally cached ``cl100k_base`` table is used through tiktoken when present.
    Minimal offline installations lack that table; in that case the documented
    UTF-8 byte heuristic is used instead of triggering tiktoken's network loader.
    """

    if not text:
        return 0
    encoding = _cached_encoding()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def tokenizer_backend() -> str:
    """Name the active approximate counting backend for diagnostics."""

    return "tiktoken:cl100k_base" if _cached_encoding() is not None else "utf8-bytes/4 fallback"


def build_outline(markdown: str, *, has_pagination: bool = False) -> Outline:
    """Build an ordered heading tree represented by section levels and anchors."""

    chunks = _section_chunks(markdown)
    return Outline(
        sections=[chunk.section for chunk in chunks],
        total_token_count=count_tokens(markdown),
        has_tables=bool(_TABLE_SEPARATOR_RE.search(markdown)),
        has_code=bool(re.search(r"^\s*(`{3,}|~{3,})", markdown, re.MULTILINE)),
        has_pagination=has_pagination or bool(_PAGINATION_RE.search(markdown)),
    )


def render_outline_head(outline: Outline) -> str:
    """Render the compact, human-readable outline-first document head."""

    flags = [
        name
        for name, present in (
            ("tables", outline.has_tables),
            ("code", outline.has_code),
            ("pagination", outline.has_pagination),
        )
        if present
    ]
    lines = [
        "# Outline",
        "",
        f"Approximate tokens: {outline.total_token_count}",
        f"Content flags: {', '.join(flags) if flags else 'none'}",
        "",
    ]
    if not outline.sections:
        lines.append("(No sections extracted)")
    for section in outline.sections:
        indent = "  " * max(section.level - 1, 0)
        lines.append(
            f"{indent}- {section.heading} [#{section.anchor}] "
            f"({section.token_count} approx. tokens, {section.char_count} chars)"
        )
    return "\n".join(lines)


def truncate_to_budget(markdown: str, budget: int | None) -> TruncatedContent:
    """Keep a prefix of whole sections and name every omitted anchor."""

    if budget is None:
        return TruncatedContent(content=markdown, truncated=False, omitted=[])
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise BudgetError("Token budget must be a non-negative integer")
    if count_tokens(markdown) <= budget:
        return TruncatedContent(content=markdown, truncated=False, omitted=[])

    chunks = _section_chunks(markdown)
    kept: list[str] = []
    used = 0
    first_omitted = len(chunks)
    for index, chunk in enumerate(chunks):
        if used + chunk.section.token_count > budget:
            first_omitted = index
            break
        kept.append(chunk.markdown)
        used += chunk.section.token_count

    omitted = [chunk.section.anchor for chunk in chunks[first_omitted:]]
    return TruncatedContent(
        content="".join(kept).rstrip(),
        truncated=True,
        omitted=omitted,
    )


def cost_for_omitted_sections(
    outline: Outline,
    omitted: list[str],
    *,
    required_tier: int | None = None,
    estimated_latency_ms: int | None = None,
) -> CostToComplete:
    """Compute a concrete completion cost from the full pre-truncation outline."""

    token_counts = {section.anchor: section.token_count for section in outline.sections}
    unknown = [anchor for anchor in omitted if anchor not in token_counts]
    if unknown:
        raise BudgetError(f"Omitted anchors are absent from the full outline: {unknown}")
    return CostToComplete(
        remaining_items=len(omitted),
        estimated_extra_tokens=sum(token_counts[anchor] for anchor in omitted),
        required_tier=required_tier,
        estimated_latency_ms=estimated_latency_ms,
    )


def section_by_anchor(markdown: str, anchor: str) -> str:
    """Return one complete Markdown section by its stable outline anchor."""

    for chunk in _section_chunks(markdown):
        if chunk.section.anchor == anchor:
            return chunk.markdown.rstrip()
    raise KeyError(anchor)


def _section_chunks(markdown: str) -> list[_SectionChunk]:
    if not markdown:
        return []

    starts: list[tuple[int, int, str]] = []
    active_fence: str | None = None
    offset = 0
    for line in markdown.splitlines(keepends=True):
        fence = MD_FENCE_RE.match(line)
        if fence:
            marker_char = fence.group(1)[0]
            if active_fence is None:
                active_fence = marker_char
            elif active_fence == marker_char:
                active_fence = None
        elif active_fence is None and (heading := MD_HEADING_RE.match(line)):
            starts.append((offset, len(heading.group(1)), heading.group(2).strip()))
        offset += len(line)

    raw_chunks: list[tuple[str, int, str]] = []
    if not starts:
        raw_chunks.append((markdown, 0, "Document"))
    else:
        first_start = starts[0][0]
        if markdown[:first_start].strip():
            raw_chunks.append((markdown[:first_start], 0, "Document"))
        for index, (start, level, heading) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(markdown)
            raw_chunks.append((markdown[start:end], level, heading))

    seen: dict[str, int] = {}
    chunks: list[_SectionChunk] = []
    for content, level, heading in raw_chunks:
        base_anchor = _slugify(heading)
        occurrence = seen.get(base_anchor, 0) + 1
        seen[base_anchor] = occurrence
        anchor = base_anchor if occurrence == 1 else f"{base_anchor}-{occurrence}"
        section = Section(
            heading=heading,
            level=level,
            anchor=anchor,
            token_count=count_tokens(content),
            char_count=len(content),
        )
        chunks.append(_SectionChunk(section=section, markdown=content))
    return chunks


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[`*_~\[\]()]", "", normalized).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "section"


@lru_cache(maxsize=1)
def _cached_encoding() -> Encoding | None:
    """Load cl100k only when its hash-verified table is already on disk."""

    if not _cl100k_cache_ready():
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except (OSError, ValueError):
        return None


def _cl100k_cache_ready() -> bool:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = str(Path(tempfile.gettempdir()) / "data-gym-cache")
    if not cache_dir:
        return False
    cache_key = hashlib.sha1(_CL100K_URL.encode()).hexdigest()
    cache_path = Path(cache_dir) / cache_key
    try:
        payload = cache_path.read_bytes()
    except OSError:
        return False
    return hashlib.sha256(payload).hexdigest() == _CL100K_SHA256
