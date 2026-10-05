"""Normalize scholarly URLs and identifiers without guessing an artifact."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
from urllib.parse import unquote, urlsplit

from selectolax.parser import HTMLParser

IdentifierKind = Literal["doi", "arxiv", "pmid", "pmcid"]

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?#]+", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)"
    r"(?P<id>(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5}))"
    r"(?:v\d+)?(?:\.pdf)?(?:[/?#].*)?$",
    re.IGNORECASE,
)
_PMCID_RE = re.compile(r"\bPMC(?P<id>\d+)\b", re.IGNORECASE)
_PMID_LABEL_RE = re.compile(r"\bPMID\s*:?\s*(?P<id>\d{4,10})\b", re.IGNORECASE)

SCHOLARLY_HOSTS = frozenset(
    {
        "nature.com",
        "sciencedirect.com",
        "linkinghub.elsevier.com",
        "cell.com",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "academic.oup.com",
        "tandfonline.com",
        "sagepub.com",
        "journals.plos.org",
        "frontiersin.org",
        "mdpi.com",
        "elifesciences.org",
        "bmj.com",
        "nejm.org",
        "thelancet.com",
        "jamanetwork.com",
        "pnas.org",
        "science.org",
        "ieeexplore.ieee.org",
        "dl.acm.org",
        "jstor.org",
        "biorxiv.org",
        "medrxiv.org",
        "arxiv.org",
        "ssrn.com",
        "osf.io",
        "zenodo.org",
        "researchgate.net",
        "semanticscholar.org",
        "openalex.org",
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "europepmc.org",
        "hal.science",
        "chemrxiv.org",
        "psyarxiv.com",
        "doi.org",
        "hdl.handle.net",
    }
)


# Route a registrable domain only through explicitly scholarly subdomains. This
# keeps aliases such as ``dx.doi.org`` and the current PMC host equivalent to
# their canonical forms without reopening broad suffix matches such as
# ``blast.ncbi.nlm.nih.gov``.
_HOST_ROUTES: tuple[tuple[str, dict[str, str]], ...] = (
    ("ncbi.nlm.nih.gov", {"": "ncbi.nlm.nih.gov", "pubmed": "pubmed.ncbi.nlm.nih.gov", "pmc": "pmc.ncbi.nlm.nih.gov"}),
    ("elsevier.com", {"linkinghub": "linkinghub.elsevier.com"}),
    ("springer.com", {"link": "link.springer.com"}),
    ("wiley.com", {"onlinelibrary": "onlinelibrary.wiley.com"}),
    ("oup.com", {"academic": "academic.oup.com"}),
    ("plos.org", {"journals": "journals.plos.org"}),
    ("acm.org", {"dl": "dl.acm.org"}),
    ("doi.org", {"": "doi.org", "dx": "doi.org"}),
    ("handle.net", {"hdl": "hdl.handle.net"}),
    ("nature.com", {"": "nature.com"}),
    ("sciencedirect.com", {"": "sciencedirect.com"}),
    ("cell.com", {"": "cell.com"}),
    ("tandfonline.com", {"": "tandfonline.com"}),
    ("sagepub.com", {"": "sagepub.com"}),
    ("frontiersin.org", {"": "frontiersin.org"}),
    ("mdpi.com", {"": "mdpi.com"}),
    ("elifesciences.org", {"": "elifesciences.org"}),
    # BMJ publishes every journal on its own subdomain (bjsm, gut, heart, jnnp,
    # thorax, ard, ...). Enumerating them goes stale, and a missing one silently
    # demotes a paper to the generic recipe, which never runs the OA ladder. The
    # ``^/content/`` path guard below is what keeps blogs.bmj.com out.
    ("bmj.com", {"": "bmj.com", "*": "bmj.com"}),
    ("nejm.org", {"": "nejm.org"}),
    ("thelancet.com", {"": "thelancet.com"}),
    ("jamanetwork.com", {"": "jamanetwork.com"}),
    ("pnas.org", {"": "pnas.org"}),
    ("science.org", {"": "science.org"}),
    ("ieeexplore.ieee.org", {"": "ieeexplore.ieee.org"}),
    ("jstor.org", {"": "jstor.org"}),
    ("biorxiv.org", {"": "biorxiv.org"}),
    ("medrxiv.org", {"": "medrxiv.org"}),
    ("arxiv.org", {"": "arxiv.org"}),
    ("ssrn.com", {"": "ssrn.com"}),
    ("osf.io", {"": "osf.io"}),
    ("zenodo.org", {"": "zenodo.org"}),
    ("researchgate.net", {"": "researchgate.net"}),
    ("semanticscholar.org", {"": "semanticscholar.org"}),
    ("openalex.org", {"": "openalex.org"}),
    ("europepmc.org", {"": "europepmc.org"}),
    ("hal.science", {"": "hal.science"}),
    ("chemrxiv.org", {"": "chemrxiv.org"}),
    ("psyarxiv.com", {"": "psyarxiv.com"}),
)

# Owning a scholarly domain is not enough: many of the same sites also serve
# news, journals, profiles, searches, and dashboards. Positive article-shaped
# paths fail closed when the route is ambiguous.
_ARTICLE_PATHS = {
    "academic.oup.com": (r"^/[^/]+/article(?:/|$)",),
    "arxiv.org": (r"^/(?:abs|pdf|html)/[^/]+",),
    "biorxiv.org": (r"^/content/10\.\d{4,9}/",),
    "bmj.com": (r"^/content/",),
    "cell.com": (r"^/[^/]+/(?:fulltext|abstract|pdf)/",),
    "chemrxiv.org": (r"^/engage/chemrxiv/article-details/",),
    "dl.acm.org": (r"^/doi/",),
    "doi.org": (r"^/10\.\d{4,9}/",),
    "elifesciences.org": (r"^/articles/",),
    "europepmc.org": (
        r"^/articles/PMC\d+",
        r"^/(?:article|abstract)/(?:MED|PMC)/",
    ),
    "frontiersin.org": (r"^/journals/[^/]+/articles/",),
    "hal.science": (r"^/hal-\d+",),
    "hdl.handle.net": (r"^/10\.\d{4,9}/",),
    "ieeexplore.ieee.org": (r"^/document/\d+",),
    "jamanetwork.com": (r"^/journals/[^/]+/(?:fullarticle|articleabstract)/",),
    "journals.plos.org": (r"^/[^/]+/article(?:/|$)",),
    "jstor.org": (r"^/stable/",),
    "link.springer.com": (r"^/article/",),
    "linkinghub.elsevier.com": (r"^/retrieve/pii/",),
    "mdpi.com": (r"^/\d{4}-\d{3,4}/\d+/\d+/\d+",),
    "medrxiv.org": (r"^/content/10\.\d{4,9}/",),
    "nature.com": (r"^/articles/",),
    "ncbi.nlm.nih.gov": (r"^/pmc/articles/PMC\d+",),
    "nejm.org": (r"^/doi/",),
    "onlinelibrary.wiley.com": (r"^/doi/",),
    "openalex.org": (r"^/works/",),
    "osf.io": (r"^/preprints/",),
    "pmc.ncbi.nlm.nih.gov": (r"^/articles/PMC\d+",),
    "pnas.org": (r"^/doi/",),
    "psyarxiv.com": (r"^/[A-Za-z0-9][A-Za-z0-9._-]{4,}/?$",),
    "pubmed.ncbi.nlm.nih.gov": (r"^/\d{4,10}/?$",),
    "researchgate.net": (r"^/publication/",),
    "sagepub.com": (r"^/doi/",),
    "science.org": (r"^/doi/",),
    "sciencedirect.com": (r"^/science/article/pii/",),
    "semanticscholar.org": (r"^/paper/",),
    "ssrn.com": (r"^/abstract(?:=|/)",),
    "tandfonline.com": (r"^/doi/",),
    "thelancet.com": (r"^/journals/[^/]+/article/",),
    "zenodo.org": (r"^/records?/",),
}


@dataclass(frozen=True, slots=True)
class ScholarlyIdentifier:
    """One canonical scholarly identifier recovered from an input."""

    kind: IdentifierKind
    value: str

    @property
    def canonical_url(self) -> str:
        """Return the public canonical resolver URL for this identifier."""

        if self.kind == "doi":
            return f"https://doi.org/{self.value}"
        if self.kind == "arxiv":
            return f"https://arxiv.org/abs/{self.value}"
        if self.kind == "pmcid":
            return f"https://europepmc.org/articles/{self.value}"
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.value}/"


def is_scholarly_input(value: str) -> bool:
    """Return whether an input is a supported identifier or scholarly URL."""

    candidate = unquote(value.strip())
    # A bare run of digits is a PMID only when the caller explicitly says so.
    # Recipe matching must never claim "12345" and route an arbitrary string
    # into the paper ladder, so the numeric form is off for discovery.
    if _bare_identifier(candidate, allow_numeric_pmid=False) is not None:
        return True
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    route = _scholarly_route(host)
    if route is None:
        return False
    path = parsed.path.rstrip("/")
    if not path:
        return False
    return any(
        re.search(pattern, path, flags=re.IGNORECASE)
        for pattern in _ARTICLE_PATHS[route]
    )


def normalize_identifier(
    value: str,
    *,
    page_html: str | None = None,
) -> ScholarlyIdentifier:
    """Normalize DOI, arXiv, PubMed and PMC inputs.

    Publisher URLs frequently carry no identifier in their path. In that case the
    fetched page must provide a ``citation_doi`` or ``DC.Identifier`` meta tag;
    the function refuses to invent an identity when neither is present.
    """

    candidate = unquote(value.strip())
    direct = _bare_identifier(candidate)
    if direct is not None:
        return direct

    if match := _ARXIV_RE.search(candidate):
        return ScholarlyIdentifier("arxiv", match.group("id"))
    if match := _PMCID_RE.search(candidate):
        return ScholarlyIdentifier("pmcid", f"PMC{match.group('id')}")

    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".").removeprefix("www.")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scholarly URL {value!r}: {exc}") from exc
    if host == "pubmed.ncbi.nlm.nih.gov":
        if match := re.search(r"/(?P<id>\d{4,10})(?:/|$)", parsed.path):
            return ScholarlyIdentifier("pmid", match.group("id"))
    if host.endswith("europepmc.org"):
        if match := re.search(r"/(?:article/MED|abstract/MED)/(?P<id>\d+)", parsed.path, re.I):
            return ScholarlyIdentifier("pmid", match.group("id"))

    if match := _DOI_RE.search(candidate):
        return ScholarlyIdentifier("doi", _clean_doi(match.group(0)))

    if page_html:
        doi = extract_meta_doi(page_html)
        if doi:
            return ScholarlyIdentifier("doi", doi)
    raise ValueError(f"No canonical scholarly identifier was found for {value!r}")


def extract_meta_doi(html: str) -> str | None:
    """Return a DOI from standard scholarly meta tags, if one is present."""

    tree = HTMLParser(html)
    for node in tree.css("meta[name]"):
        name = (node.attributes.get("name") or "").strip().lower()
        if name not in {"citation_doi", "dc.identifier", "dc.identifier.doi"}:
            continue
        content = unquote((node.attributes.get("content") or "").strip())
        if match := _DOI_RE.search(content):
            return _clean_doi(match.group(0))
    return None


def _bare_identifier(
    value: str,
    *,
    allow_numeric_pmid: bool = True,
) -> ScholarlyIdentifier | None:
    stripped = value.strip()
    if match := _ARXIV_RE.fullmatch(stripped):
        return ScholarlyIdentifier("arxiv", match.group("id"))
    if match := _PMCID_RE.fullmatch(stripped):
        return ScholarlyIdentifier("pmcid", f"PMC{match.group('id')}")
    if match := _PMID_LABEL_RE.fullmatch(stripped):
        return ScholarlyIdentifier("pmid", match.group("id"))
    if allow_numeric_pmid and stripped.isdigit() and 4 <= len(stripped) <= 10:
        return ScholarlyIdentifier("pmid", stripped)
    if match := _DOI_RE.fullmatch(stripped):
        return ScholarlyIdentifier("doi", _clean_doi(match.group(0)))
    return None


def _clean_doi(value: str) -> str:
    return value.rstrip(".,;:)]}").lower()


def _scholarly_route(host: str) -> str | None:
    normalized = host.casefold().rstrip(".")
    if normalized.startswith("www."):
        normalized = normalized[4:]
    for domain, routes in _HOST_ROUTES:
        if normalized == domain:
            subdomain = ""
        elif normalized.endswith(f".{domain}"):
            subdomain = normalized[: -(len(domain) + 1)]
        else:
            continue
        return routes.get(subdomain) or routes.get("*")
    return None
