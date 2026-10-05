"""Artifact identity checks proportional to locator provenance."""

from __future__ import annotations

import re

from .pdf import pdf_reader
from selectolax.parser import HTMLParser

from ..types import DependencyError
from .metadata import ScholarMetadata

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "via",
        "with",
    }
)


def verify_identity(
    payload: bytes,
    kind: str,
    metadata: ScholarMetadata,
    *,
    bound: bool,
) -> tuple[bool, str]:
    """Return whether downloaded bytes plausibly belong to the requested work.

    Bound locators come from a work-specific mapping, so an available author is the
    sanity check and title drift is tolerated. Scraped or otherwise unbound locators
    must prove both title and author in their front matter.
    """

    front = _front_text(payload, kind)[:8_000]
    if not front.strip():
        if bound:
            return True, "No front matter was extractable; accepted on bound provenance"
        return False, "No front matter was extractable for an unbound locator"

    front_name_tokens = set(re.findall(r"[a-z0-9]+", front.casefold()))
    author_ok = any(
        bool(parts := re.findall(r"[a-z0-9]+", author.casefold()))
        and all(part in front_name_tokens for part in parts)
        for author in metadata.authors
    )
    title_tokens = _tokens(metadata.title or "")
    title_overlap = (
        len(title_tokens & _tokens(front)) / len(title_tokens)
        if title_tokens
        else 0.0
    )
    if bound:
        if metadata.authors and not author_ok and title_overlap < 0.4:
            return False, (
                "Bound locator matched neither an author nor enough title tokens "
                f"({title_overlap:.0%})"
            )
        return True, (
            "Bound locator passed the author sanity check"
            if author_ok
            else (
                f"Bound locator accepted on provenance and {title_overlap:.0%} title overlap"
                if metadata.authors
                else "Bound locator accepted because no author manifest was available"
            )
        )

    if not metadata.authors:
        return False, "Unbound locator cannot be verified without an author manifest"
    if not author_ok:
        return False, "No manifest author appears in the artifact front matter"
    if not title_tokens:
        return False, "Unbound locator cannot be verified without a title manifest"
    if title_overlap < 0.6:
        return False, f"Artifact title overlap is only {title_overlap:.0%}"
    return (
        True,
        f"Artifact front matter matches an author and {title_overlap:.0%} of title tokens",
    )


def _front_text(payload: bytes, kind: str) -> str:
    if kind == "pdf":
        try:
            reader = pdf_reader(payload)
            return "\n".join(
                (page.extract_text() or "").strip() for page in reader.pages[:2]
            )
        except DependencyError:
            raise
        except Exception:
            return ""
    text = payload.decode("utf-8", errors="replace")
    if kind == "html":
        tree = HTMLParser(text[:250_000])
        root = tree.body or tree.root
        return " ".join(root.text(separator=" ").split()) if root is not None else ""
    if kind == "jats":
        tree = HTMLParser(text[:250_000])
        return " ".join(tree.text(separator=" ").split())
    return text


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }
