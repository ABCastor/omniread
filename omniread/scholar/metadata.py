"""Collect independent scholarly manifests from public metadata services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
import re
from typing import Iterable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from selectolax.parser import HTMLParser

from ..recipes.base import RecipeHttpClient
from ..types import JsonValue, RetrievalError
from .http import get_with_retry
from .ids import ScholarlyIdentifier


@dataclass(frozen=True, slots=True)
class OALocation:
    """One open-access locator claim, not proof that bytes are reachable."""

    url: str
    kind: str
    provider: str


def strip_contact_query(url: str) -> str:
    """Remove contact-address parameters from any URL exposed to callers."""

    try:
        parts = urlsplit(url)
    except (TypeError, ValueError):
        return url
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() not in {"email", "mailto"}
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment)
    )


@dataclass(slots=True)
class ScholarMetadata:
    """Merged manifests with source-specific counts kept separate."""

    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    authors: list[str] = field(default_factory=list)
    first_page: str | None = None
    last_page: str | None = None
    crossref_reference_count: int | None = None
    openalex_reference_count: int | None = None
    oa_locations: list[OALocation] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    def merge(self, other: "ScholarMetadata") -> None:
        """Merge one source without replacing already-observed values."""

        for name in (
            "doi",
            "title",
            "abstract",
            "pmid",
            "pmcid",
            "arxiv_id",
            "first_page",
            "last_page",
            "crossref_reference_count",
            "openalex_reference_count",
        ):
            if getattr(self, name) is None and getattr(other, name) is not None:
                setattr(self, name, getattr(other, name))
        known = {(item.url, item.kind, item.provider) for item in self.oa_locations}
        for item in other.oa_locations:
            key = (item.url, item.kind, item.provider)
            if key not in known:
                self.oa_locations.append(item)
                known.add(key)
        self.authors = list(dict.fromkeys((*self.authors, *other.authors)))
        self.source_urls = list(dict.fromkeys((*self.source_urls, *other.source_urls)))
        self.failures.update(other.failures)

    def as_json(self) -> dict[str, JsonValue]:
        """Return the manifest in the shared JSON-compatible shape."""

        return {
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "arxiv_id": self.arxiv_id,
            "authors": list(self.authors),
            "biblio": {
                "first_page": self.first_page,
                "last_page": self.last_page,
            },
            "reference_counts": {
                "crossref": self.crossref_reference_count,
                "openalex": self.openalex_reference_count,
            },
            "oa_locations": [
                {
                    "url": strip_contact_query(item.url),
                    "kind": item.kind,
                    "provider": item.provider,
                }
                for item in self.oa_locations
            ],
            "metadata_failures": dict(self.failures),
        }


def collect_metadata(
    identifier: ScholarlyIdentifier,
    http: RecipeHttpClient,
) -> ScholarMetadata:
    """Fetch OpenAlex, Crossref and Europe PMC independently.

    A dead or malformed source is recorded and skipped. It cannot turn weak
    evidence into a stronger verdict.
    """

    merged = ScholarMetadata()
    if identifier.kind == "doi":
        merged.doi = identifier.value
    elif identifier.kind == "pmid":
        merged.pmid = identifier.value
    elif identifier.kind == "pmcid":
        merged.pmcid = identifier.value
    else:
        merged.arxiv_id = identifier.value

    query = _europe_pmc_query(identifier)
    epmc_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={quote(query, safe=':')}&format=json"
    )
    if identifier.kind == "doi":
        _collect_doi_manifests(identifier.value, http, merged)
        _collect_source(
            "europe_pmc",
            epmc_url,
            lambda value: parse_europe_pmc(value, expected=identifier),
            http,
            merged,
        )
    else:
        # PubMed/PMC inputs reveal their DOI through Europe PMC. Once known, use it
        # to collect the two independent DOI manifests too.
        _collect_source(
            "europe_pmc",
            epmc_url,
            lambda value: parse_europe_pmc(value, expected=identifier),
            http,
            merged,
        )
        if merged.doi:
            _collect_doi_manifests(merged.doi, http, merged)
    return merged


def _collect_doi_manifests(
    doi: str,
    http: RecipeHttpClient,
    merged: ScholarMetadata,
) -> None:
    openalex_doi = quote(doi, safe="/")
    crossref_doi = quote(doi, safe="")
    contact = os.environ.get("OMNIREAD_CONTACT_EMAIL", "").strip()
    suffix = f"?mailto={quote(contact, safe='@')}" if contact else ""
    _collect_source(
        "openalex",
        f"https://api.openalex.org/works/doi:{openalex_doi}{suffix}",
        parse_openalex,
        http,
        merged,
    )
    _collect_source(
        "crossref",
        f"https://api.crossref.org/works/{crossref_doi}",
        parse_crossref,
        http,
        merged,
    )


def _collect_source(
    name: str,
    url: str,
    parser: Callable[[dict[str, JsonValue]], ScholarMetadata],
    http: RecipeHttpClient,
    merged: ScholarMetadata,
) -> None:
    try:
        response = get_with_retry(http, url)
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("metadata response is not a JSON object")
        parsed = parser(value)
        parsed.source_urls.append(strip_contact_query(response.final_url))
        merged.merge(parsed)
    except (RetrievalError, TypeError, ValueError) as exc:
        merged.failures[name] = str(exc)


def parse_openalex(value: dict[str, JsonValue]) -> ScholarMetadata:
    """Parse an OpenAlex work response, including every OA location."""

    result = ScholarMetadata(
        doi=_doi_value(value.get("doi")),
        title=_string(value.get("title")) or _string(value.get("display_name")),
        abstract=reconstruct_abstract(value.get("abstract_inverted_index")),
        openalex_reference_count=_integer(value.get("referenced_works_count")),
        authors=_openalex_authors(value.get("authorships")),
    )
    biblio = value.get("biblio")
    if isinstance(biblio, dict):
        result.first_page = _string(biblio.get("first_page"))
        result.last_page = _string(biblio.get("last_page"))
    ids = value.get("ids")
    if isinstance(ids, dict):
        result.pmid = _trailing_identifier(ids.get("pmid"), "pmid")
        result.pmcid = _trailing_identifier(ids.get("pmcid"), "pmc")
        result.arxiv_id = _trailing_identifier(ids.get("arxiv"), "arxiv")

    raw_locations: list[JsonValue] = []
    best = value.get("best_oa_location")
    if isinstance(best, dict):
        raw_locations.append(best)
    locations = value.get("locations")
    if isinstance(locations, list):
        raw_locations.extend(locations)
    open_access = value.get("open_access")
    if isinstance(open_access, dict) and isinstance(open_access.get("oa_url"), str):
        result.oa_locations.append(
            OALocation(str(open_access["oa_url"]), "landing", "openalex")
        )
    for location in raw_locations:
        if not isinstance(location, dict):
            continue
        provider = "openalex"
        source = location.get("source")
        if isinstance(source, dict) and isinstance(source.get("type"), str):
            provider = f"openalex:{source['type']}"
        pdf = location.get("pdf_url")
        landing = location.get("landing_page_url")
        if isinstance(pdf, str) and pdf:
            result.oa_locations.append(OALocation(pdf, "pdf", provider))
        if isinstance(landing, str) and landing:
            result.oa_locations.append(OALocation(landing, "landing", provider))
    result.oa_locations = _deduplicate_locations(result.oa_locations)
    return result


def parse_crossref(value: dict[str, JsonValue]) -> ScholarMetadata:
    """Parse the Crossref ``message`` manifest."""

    message = value.get("message")
    if not isinstance(message, dict):
        raise ValueError("Crossref returned an invalid metadata envelope")
    title_value = message.get("title")
    title = (
        _string(title_value[0])
        if isinstance(title_value, list) and title_value
        else _string(title_value)
    )
    first_page, last_page = _page_range(_string(message.get("page")))
    return ScholarMetadata(
        doi=_doi_value(message.get("DOI")),
        title=title,
        abstract=_clean_markup(_string(message.get("abstract"))),
        first_page=first_page,
        last_page=last_page,
        crossref_reference_count=_integer(message.get("reference-count")),
        authors=_crossref_authors(message.get("author")),
    )


def parse_europe_pmc(
    value: dict[str, JsonValue],
    *,
    expected: ScholarlyIdentifier | None = None,
) -> ScholarMetadata:
    """Parse the matching Europe PMC search result."""

    result_list = value.get("resultList")
    rows = result_list.get("result") if isinstance(result_list, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Europe PMC returned no result list")
    row = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and _matches_identifier(item, expected)
        ),
        None,
    )
    if not isinstance(row, dict):
        raise ValueError("Europe PMC returned no matching scholarly record")
    first_page, last_page = _page_range(
        _string(row.get("pageInfo")) or _string(row.get("page"))
    )
    metadata = ScholarMetadata(
        doi=_doi_value(row.get("doi")),
        title=_string(row.get("title")),
        abstract=_clean_markup(
            _string(row.get("abstractText")) or _string(row.get("abstract"))
        ),
        pmid=_string(row.get("pmid")),
        pmcid=_normalized_pmcid(row.get("pmcid")),
        first_page=first_page,
        last_page=last_page,
    )
    full_text_urls = row.get("fullTextUrlList")
    urls = full_text_urls.get("fullTextUrl") if isinstance(full_text_urls, dict) else None
    if isinstance(urls, list):
        for item in urls:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            style = item.get("documentStyle")
            if isinstance(url, str) and isinstance(style, str):
                metadata.oa_locations.append(
                    OALocation(url, style.lower(), "europe-pmc")
                )
    return metadata


def reconstruct_abstract(value: JsonValue) -> str | None:
    """Reconstruct OpenAlex's position-indexed abstract."""

    if not isinstance(value, dict) or not value:
        return None
    positions: list[tuple[int, str]] = []
    for word, raw_positions in value.items():
        if not isinstance(word, str) or not isinstance(raw_positions, list):
            continue
        positions.extend(
            (position, word)
            for position in raw_positions
            if isinstance(position, int) and position >= 0
        )
    if not positions:
        return None
    return " ".join(word for _, word in sorted(positions))


def _openalex_authors(value: JsonValue) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for authorship in value:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        name = _string(author.get("display_name")) if isinstance(author, dict) else None
        if name:
            authors.append(name.rsplit(" ", 1)[-1])
    return list(dict.fromkeys(authors))


def _crossref_authors(value: JsonValue) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for author in value:
        if not isinstance(author, dict):
            continue
        name = _string(author.get("family")) or _string(author.get("name"))
        if name:
            authors.append(name)
    return list(dict.fromkeys(authors))


def _europe_pmc_query(identifier: ScholarlyIdentifier) -> str:
    labels = {
        "doi": "DOI",
        "pmid": "EXT_ID",
        "pmcid": "PMC_ID",
        "arxiv": "ARXIV",
    }
    return f"{labels[identifier.kind]}:{identifier.value}"


def _matches_identifier(
    row: dict[str, JsonValue],
    expected: ScholarlyIdentifier | None,
) -> bool:
    if expected is None:
        return True
    if expected.kind == "doi":
        return (_doi_value(row.get("doi")) or "") == expected.value.lower()
    if expected.kind == "pmid":
        return _string(row.get("pmid")) == expected.value
    if expected.kind == "pmcid":
        return _normalized_pmcid(row.get("pmcid")) == expected.value.upper()
    arxiv = _string(row.get("arxivId")) or _string(row.get("arxiv_id"))
    return bool(arxiv and arxiv.removesuffix(".pdf") == expected.value)


def _deduplicate_locations(values: Iterable[OALocation]) -> list[OALocation]:
    seen: set[tuple[str, str]] = set()
    result: list[OALocation] = []
    for item in values:
        key = (item.url, item.kind)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _page_range(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    parts = re.split(r"\s*[-–]\s*", value, maxsplit=1)
    return parts[0] or None, parts[1] if len(parts) > 1 and parts[1] else parts[0]


def _clean_markup(value: str | None) -> str | None:
    if not value:
        return None
    text = " ".join(HTMLParser(value).text().split())
    return text or None


def _doi_value(value: JsonValue) -> str | None:
    text = _string(value)
    if not text:
        return None
    match = re.search(r"10\.\d{4,9}/\S+", text, re.IGNORECASE)
    return match.group(0).rstrip(".,;").lower() if match else None


def _trailing_identifier(value: JsonValue, label: str) -> str | None:
    text = _string(value)
    if not text:
        return None
    tail = text.rstrip("/").rsplit("/", 1)[-1]
    if label == "pmc":
        return _normalized_pmcid(tail)
    return tail.removeprefix(f"{label}:")


def _normalized_pmcid(value: JsonValue) -> str | None:
    text = _string(value)
    if not text:
        return None
    match = re.search(r"PMC\d+", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _string(value: JsonValue) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
