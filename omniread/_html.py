"""Small DOM probes shared by extraction, verification, and block detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterator
from urllib.parse import parse_qsl, urljoin, urlsplit

from selectolax.parser import HTMLParser, Node

from .types import JsonValue

_WORD_RE = re.compile(r"[\w]+(?:[’'-][\w]+)*", re.UNICODE)
# Shared Markdown line patterns: a heading line, and a fenced-code delimiter. Centralized so the
# verifier and the outline/truncation layer agree on "what is a heading" and can never drift apart
# (a divergence there would be a silent verifier correctness bug).
MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
MD_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DROP_ALWAYS = "script, style, noscript, template, svg, canvas"
_DROP_BOILERPLATE = "nav, aside, footer, form"


@dataclass(frozen=True, slots=True)
class DomSnapshot:
    """Independent visible-DOM facts used by the verifier."""

    text: str
    word_count: int
    headings: tuple[str, ...]
    title: str
    has_content_container: bool


def normalize_space(value: str) -> str:
    """Collapse whitespace without changing the words themselves."""

    return re.sub(r"\s+", " ", value).strip()


def word_count(value: str) -> int:
    """Count lexical words for relative, not tokenizer, measurements."""

    return len(_WORD_RE.findall(value))


def _attr(node: Node, name: str) -> str:
    """Return one HTML attribute as text; valueless attributes normalize to empty."""

    return node.attributes.get(name) or ""


def strip_scripts_and_styles(raw_html: str) -> str:
    """Remove non-content executable/style nodes before extraction or measuring."""

    tree = HTMLParser(raw_html)
    for node in tree.css(_DROP_ALWAYS):
        node.decompose()
    return tree.html or ""


def visible_dom_snapshot(raw_html: str) -> DomSnapshot:
    """Measure the visible main DOM independently from Trafilatura's output."""

    tree = HTMLParser(raw_html)
    title_node = tree.css_first("title")
    title = normalize_space(title_node.text(separator=" ", strip=True)) if title_node else ""

    for node in tree.css(_DROP_ALWAYS):
        node.decompose()
    for node in tree.css('[hidden], [aria-hidden="true"], [aria-hidden="True"]'):
        node.decompose()
    for node in tree.css("[style]"):
        style = re.sub(r"\s+", "", _attr(node, "style").lower())
        if "display:none" in style or "visibility:hidden" in style:
            node.decompose()

    scope = (
        tree.css_first("article")
        or tree.css_first("main")
        or tree.css_first('[role="main"]')
    )
    has_content_container = scope is not None
    scope = scope or tree.body or tree.root

    for node in scope.css(_DROP_BOILERPLATE):
        node.decompose()
    headings = tuple(
        text
        for node in scope.css("h1, h2, h3, h4, h5, h6")
        if (text := normalize_space(node.text(separator=" ", strip=True)))
    )
    text = normalize_space(scope.text(separator=" ", strip=True))
    return DomSnapshot(
        text=text,
        word_count=word_count(text),
        headings=headings,
        title=title,
        has_content_container=has_content_container,
    )


def extract_json_ld(raw_html: str) -> JsonValue:
    """Parse every valid JSON-LD block without treating it as page content."""

    tree = HTMLParser(raw_html)
    values: list[JsonValue] = []
    for node in tree.css('script[type="application/ld+json"]'):
        payload = node.text(separator="", strip=True).strip()
        payload = re.sub(r"^\s*<!--|-->\s*$", "", payload).strip()
        if not payload:
            continue
        try:
            values.append(json.loads(payload))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def canonical_url(raw_html: str, fallback_url: str) -> str:
    """Resolve the document's canonical link, falling back to the final response URL."""

    tree = HTMLParser(raw_html)
    for node in tree.css("link[rel]"):
        rel = _attr(node, "rel").lower().split()
        href = _attr(node, "href").strip()
        if "canonical" in rel and href:
            return urljoin(fallback_url, href)
    return fallback_url


def has_pagination(raw_html: str, *, current_url: str = "") -> bool:
    """Detect content pagination without confusing next-document navigation.

    A visible pagination widget is explicit evidence. A bare ``rel=next`` only
    counts when its URL is recognizably another page of the current resource.
    """

    tree = HTMLParser(raw_html)
    if any(not _is_hidden(node) for node in tree.css(".pagination, .pager")):
        return True
    for node in tree.css("[aria-label]"):
        if "pagination" in _attr(node, "aria-label").lower() and not _is_hidden(node):
            return True
    for node in tree.css("nav, [role='navigation']"):
        if _is_hidden(node):
            continue
        link_texts = [
            normalize_space(link.text(separator=" ", strip=True))
            for link in node.css("a")
            if not _is_hidden(link)
        ]
        control_words = {"first", "last", "next", "prev", "previous", "«", "‹", "›", "»"}
        numbered_links = {text for text in link_texts if text.isdigit()}
        if current_url and any(
            normalize_space(link.text(separator=" ", strip=True)).isdigit()
            and not _is_hidden(link)
            and _is_same_resource_page(current_url, _attr(link, "href"))
            for link in node.css("a[href]")
        ):
            return True
        if len(numbered_links) >= 2 and all(
            text.isdigit() or text.lower() in control_words for text in link_texts
        ):
            return True

    if not current_url:
        return False
    for node in tree.css("link[rel], a[href]"):
        rel = _attr(node, "rel").lower().split()
        href = _attr(node, "href").strip()
        labels = [
            normalize_space(node.text(separator=" ", strip=True)).lower(),
            normalize_space(_attr(node, "aria-label")).lower(),
        ]
        is_next = "next" in rel or any(
            re.match(r"^next\b", label) or label in {"›", "»", "→", "➜", "⟶"}
            for label in labels
        )
        if is_next and href and not _is_hidden(node) and _is_same_resource_page(current_url, href):
            return True
    return False


def _is_hidden(node: Node) -> bool:
    current: Node | None = node
    while current is not None:
        attrs = current.attributes
        if "hidden" in attrs or _attr(current, "aria-hidden").lower() == "true":
            return True
        style = re.sub(r"\s+", "", _attr(current, "style").lower())
        if "display:none" in style or "visibility:hidden" in style:
            return True
        current = current.parent
    return False


def _is_same_resource_page(current_url: str, href: str) -> bool:
    current = urlsplit(current_url)
    candidate = urlsplit(urljoin(current_url, href))
    if candidate.scheme not in {"http", "https"} or not candidate.netloc:
        return False
    if (candidate.scheme, candidate.netloc) != (current.scheme, current.netloc):
        return False
    if candidate.geturl() == current.geturl():
        return False

    if candidate.path == current.path and _query_diff_is_page_only(current.query, candidate.query):
        return True

    current_path = current.path.rstrip("/")
    candidate_path = candidate.path.rstrip("/")
    current_base = re.sub(r"/[1-9]\d*$", "", current_path)
    candidate_base = re.sub(r"/[1-9]\d*$", "", candidate_path)
    if (
        current_base
        and candidate_base == current_base
        and candidate_base != candidate_path
        and _same_non_page_query(current.query, candidate.query)
    ):
        return True

    page_segment = re.fullmatch(rf"{re.escape(current_path)}/page/[1-9]\d*", candidate_path, re.I)
    if page_segment and _same_non_page_query(current.query, candidate.query):
        return True

    current_suffix = re.fullmatch(r"(?P<stem>.+)(?P<ext>\.[A-Za-z0-9]+)", current_path)
    candidate_suffix = re.fullmatch(
        r"(?P<stem>.+)-(?P<page>[2-9]\d*)(?P<ext>\.[A-Za-z0-9]+)",
        candidate_path,
    )
    return bool(
        current_suffix
        and candidate_suffix
        and current_suffix.group("stem") == candidate_suffix.group("stem")
        and current_suffix.group("ext").lower() == candidate_suffix.group("ext").lower()
        and _same_non_page_query(current.query, candidate.query)
    )


_PAGE_QUERY_KEYS = frozenset({"page", "paged", "p"})


def _query_diff_is_page_only(current_query: str, candidate_query: str) -> bool:
    candidate_pairs = parse_qsl(candidate_query, keep_blank_values=True)
    candidate_pages = [value for key, value in candidate_pairs if key.lower() in _PAGE_QUERY_KEYS]
    if not candidate_pages or not all(value.isdigit() and int(value) > 0 for value in candidate_pages):
        return False
    current_pairs = parse_qsl(current_query, keep_blank_values=True)
    return _without_page_params(current_pairs) == _without_page_params(candidate_pairs)


def _same_non_page_query(current_query: str, candidate_query: str) -> bool:
    return _without_page_params(parse_qsl(current_query, keep_blank_values=True)) == _without_page_params(
        parse_qsl(candidate_query, keep_blank_values=True)
    )


def _without_page_params(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted((key, value) for key, value in pairs if key.lower() not in _PAGE_QUERY_KEYS)


def declared_word_count(raw_html: str, structured_data: JsonValue) -> int | None:
    """Read a publisher-declared word count from JSON-LD or explicit metadata."""

    candidates: list[int] = []
    article_bodies: list[int] = []
    for value in _walk_json(structured_data):
        if not isinstance(value, dict):
            continue
        raw_count = value.get("wordCount")
        parsed = _parse_positive_int(raw_count)
        if parsed is not None:
            candidates.append(parsed)
        article_body = value.get("articleBody")
        if isinstance(article_body, str) and article_body.strip():
            article_bodies.append(word_count(article_body))

    tree = HTMLParser(raw_html)
    for node in tree.css("meta[name], meta[property]"):
        key = (
            _attr(node, "name")
            or _attr(node, "property")
        ).lower()
        if key in {"wordcount", "article:word_count", "parsely-post-word-count"}:
            parsed = _parse_positive_int(_attr(node, "content"))
            if parsed is not None:
                candidates.append(parsed)

    if candidates:
        return max(candidates)
    if article_bodies:
        return max(article_bodies)
    return None


def _walk_json(value: JsonValue) -> Iterator[JsonValue]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _parse_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d[\d,._ ]*", value)
        if match:
            digits = re.sub(r"\D", "", match.group())
            if digits:
                parsed = int(digits)
                return parsed if parsed > 0 else None
    return None
