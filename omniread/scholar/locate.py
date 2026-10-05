"""Acquire scholarly full text through the evidence-ordered OA ladder."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import os
import socket
from collections.abc import Callable, Iterable
from typing import Literal
from urllib.parse import SplitResult, quote, urljoin, urlsplit

from selectolax.parser import HTMLParser

from ..recipes.base import HttpResponse, ReadOptions, RecipeContext
from ..types import JsonValue, OmniReadError, ReadResult, RetrievalError
from .http import get_with_retry
from .ids import ScholarlyIdentifier
from .metadata import OALocation, ScholarMetadata, strip_contact_query
from .xml import parse_jats_xml

AcquisitionKind = Literal["jats", "html", "pdf", "markdown"]
HostResolver = Callable[[str], Iterable[str]]
CandidateValidator = Callable[["AcquiredPaper"], tuple[bool, str]]

_MAX_CANDIDATE_BYTES = 25 * 1024 * 1024
_MAX_REDIRECT_HOPS = 3


@dataclass(frozen=True, slots=True)
class AcquisitionAttempt:
    """One content-locator attempt and its observed outcome."""

    rung: int
    provider: str
    url: str
    succeeded: bool
    detail: str

    def as_json(self) -> dict[str, JsonValue]:
        public_url = strip_contact_query(self.url)
        return {
            "rung": self.rung,
            "provider": self.provider,
            "url": public_url,
            "succeeded": self.succeeded,
            "detail": self.detail.replace(self.url, public_url),
        }


@dataclass(frozen=True, slots=True)
class AcquiredPaper:
    """Verified bytes or Markdown produced by one acquisition rung."""

    kind: AcquisitionKind
    payload: bytes
    rung: int
    provider: str
    final_url: str
    http_status: int | None
    source_urls: tuple[str, ...]
    identity_bound: bool = False
    # Set only by rungs that ran the generic ladder, which knows which tier
    # actually produced the bytes. Reporting rung 8 as Tier 4 unconditionally
    # would credit a login for a page that was read anonymously.
    tier: int | None = None


@dataclass(slots=True)
class AcquisitionResult:
    """The winning representation, or an honest record that every rung failed."""

    paper: AcquiredPaper | None = None
    attempts: list[AcquisitionAttempt] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    untried_unlocks: list[str] = field(default_factory=list)
    last_resort_pdf_path: str | None = None


def acquire_full_text(
    *,
    identifier: ScholarlyIdentifier,
    metadata: ScholarMetadata,
    original_url: str,
    landing: HttpResponse | None,
    opts: ReadOptions,
    ctx: RecipeContext,
    resolve: HostResolver | None = None,
    validate: CandidateValidator | None = None,
) -> AcquisitionResult:
    """Run ADR 0006's nine rungs in their measured order."""

    resolver = resolve or resolve_host
    result = AcquisitionResult()

    # 1. Europe PMC JATS
    pmcid = metadata.pmcid or (
        identifier.value if identifier.kind == "pmcid" else None
    )
    if pmcid:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/"
            f"{pmcid}/fullTextXML"
        )
        paper = _attempt_candidate(
            result,
            ctx,
            rung=1,
            provider="europe-pmc-jats",
            url=url,
            expected="jats",
            resolve=resolver,
            identity_bound=True,
        )
        if paper and (winner := _accept(result, paper, validate)):
            return winner

        # Europe PMC's XML endpoint covers only a subset of readable PMC papers.
        # A resolved PMCID still binds the PMC artifact to this DOI, so try PMC's
        # preferred PDF route and then its substantial article page.
        for provider, direct_url, expected, minimum in (
            (
                "pmc-direct-pdf",
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/",
                "pdf",
                None,
            ),
            (
                "pmc-direct-html",
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                "html",
                15_000,
            ),
        ):
            paper = _attempt_candidate(
                result,
                ctx,
                rung=1,
                provider=provider,
                url=direct_url,
                expected=expected,
                resolve=resolver,
                identity_bound=True,
                minimum_visible_chars=minimum,
            )
            if paper and (winner := _accept(result, paper, validate)):
                return winner

    # 2. bioRxiv / medRxiv JATS
    doi = metadata.doi or (identifier.value if identifier.kind == "doi" else None)
    if doi:
        preferred = "medrxiv" if _host_has(original_url, "medrxiv.org") else "biorxiv"
        direct_servers = (preferred, "medrxiv" if preferred == "biorxiv" else "biorxiv")
        preprints = (
            [(server, doi) for server in direct_servers]
            if doi.startswith("10.1101/")
            else _published_preprints(doi, result, ctx)
        )
        for server, preprint_doi in preprints:
            api_url = (
                f"https://api.biorxiv.org/details/{server}/"
                f"{quote(preprint_doi, safe='/')}"
            )
            jats_url = _biorxiv_jats_url(api_url, result, ctx)
            if not jats_url:
                continue
            paper = _attempt_candidate(
                result,
                ctx,
                rung=2,
                provider=f"{server}-jats",
                url=jats_url,
                expected="jats",
                resolve=resolver,
                identity_bound=True,
            )
            if paper and (winner := _accept(result, paper, validate)):
                return winner

    # 3. arXiv HTML, then ar5iv.
    arxiv_id = metadata.arxiv_id or (
        identifier.value if identifier.kind == "arxiv" else _arxiv_from_doi(doi)
    )
    if arxiv_id:
        for provider, url in (
            ("arxiv-html", f"https://arxiv.org/html/{arxiv_id}"),
            ("ar5iv-html", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"),
        ):
            paper = _attempt_candidate(
                result,
                ctx,
                rung=3,
                provider=provider,
                url=url,
                expected="html",
                resolve=resolver,
                identity_bound=True,
            )
            if paper and (winner := _accept(result, paper, validate)):
                return winner

    # 4. OpenAlex PDFs.
    for location in _locations(metadata.oa_locations, provider_prefix="openalex", kind="pdf"):
        paper = _attempt_candidate(
            result,
            ctx,
            rung=4,
            provider=location.provider,
            url=location.url,
            expected="pdf",
            resolve=resolver,
            identity_bound=True,
        )
        if paper and (winner := _accept(result, paper, validate)):
            return winner

    # 5. Unpaywall PDFs.
    unpaywall_locations: list[OALocation] = []
    if doi:
        unpaywall_locations = _fetch_unpaywall(doi, result, ctx)
        for location in _locations(unpaywall_locations, kind="pdf"):
            paper = _attempt_candidate(
                result,
                ctx,
                rung=5,
                provider=location.provider,
                url=location.url,
                expected="pdf",
                resolve=resolver,
                identity_bound=True,
            )
            if paper and (winner := _accept(result, paper, validate)):
                return winner

    # 6. The publisher/landing page's citation_pdf_url.
    if landing:
        for candidate in citation_pdf_urls(landing.text, landing.final_url):
            paper = _attempt_candidate(
                result,
                ctx,
                rung=6,
                provider="landing-citation-pdf",
                url=candidate,
                expected="pdf",
                resolve=resolver,
                identity_bound=False,
            )
            if paper and (winner := _accept(result, paper, validate)):
                return winner

    # 7. Repository landing page -> its own advertised PDF, exactly one hop.
    publisher_host = _hostname(landing.final_url if landing else original_url) or ""
    landings = [
        *_locations(metadata.oa_locations, kind="landing"),
        *_locations(unpaywall_locations, kind="landing"),
    ]
    for location in _deduplicate_locations(landings):
        host = _hostname(location.url)
        if host is None:
            result.attempts.append(
                AcquisitionAttempt(
                    7,
                    f"{location.provider}-landing",
                    location.url,
                    False,
                    "Rejected malformed repository landing URL before fetch",
                )
            )
            continue
        if not host or host == publisher_host or host in {"doi.org", "pubmed.ncbi.nlm.nih.gov"}:
            continue
        repository = _fetch_response(
            result,
            ctx,
            rung=7,
            provider=f"{location.provider}-landing",
            url=location.url,
            resolve=resolver,
        )
        if repository is None:
            continue
        for candidate in citation_pdf_urls(repository.text, repository.final_url):
            paper = _attempt_candidate(
                result,
                ctx,
                rung=7,
                provider=f"{location.provider}-one-hop-pdf",
                url=candidate,
                expected="pdf",
                resolve=resolver,
                identity_bound=False,
            )
            if paper and (winner := _accept(result, paper, validate)):
                return winner

    # 8. The publisher's own article page: anonymously first, then with a stored login.
    #
    # A paper can be free to read at the publisher without being open access
    # anywhere. BMJ marks consensus statements FREE on their own site while
    # Unpaywall, OpenAlex, Crossref and Europe PMC all still report "subscription
    # required", so every rung above — each keyed on an OA index or on the URL the
    # caller happened to hand us — reports nothing and the paper is declared
    # paywalled while a browser reads it for free. Reading the publisher page is
    # the only rung that can see that, and it is also the first rung that sends a
    # browser, which is what a bot challenge is asking for.
    publisher_url = _publisher_read_target(
        landing=landing,
        metadata=metadata,
        original_url=original_url,
        doi=doi,
    )
    observed_publisher: str | None = None
    if publisher_url is not None:
        paper, observed_publisher = _read_publisher_page(
            result,
            ctx,
            url=publisher_url,
            authenticated=False,
        )
        if paper and (winner := _accept(result, paper, validate)):
            return winner

    auth_target = observed_publisher or _publisher_target(
        landing=landing,
        metadata=metadata,
        original_url=original_url,
    )
    if auth_target is None:
        result.untried_unlocks.append(
            "resolve the publisher page for this DOI, then store a Tier-4 profile for it"
        )
    elif ctx.profiles.exists(auth_target):
        paper, _ = _read_publisher_page(
            result,
            ctx,
            url=auth_target,
            authenticated=True,
        )
        if paper and (winner := _accept(result, paper, validate)):
            return winner
    else:
        result.untried_unlocks.append(
            f"omniread login {ctx.profiles.domain(auth_target)}"
        )

    # 9. Optional last-resort acquirer, supplied by the operator.
    # The published core ships no implementation: this hook is None unless a
    # local plugin is installed (see reader._load_last_resort_acquirer). Whatever
    # it returns is treated as bound provenance and still passes the same
    # artifact identity gate as every other rung.
    last_resort_pdf_path: str | None = None
    if doi and ctx.last_resort is not None:
        outcome = ctx.last_resort.acquire(doi, resolve=resolver)
        for item in outcome.attempts:
            result.attempts.append(
                AcquisitionAttempt(
                    9,
                    item.provider,
                    item.url,
                    item.succeeded,
                    item.detail,
                )
            )
        if outcome.paper is not None:
            paper = AcquiredPaper(
                kind="pdf",
                payload=outcome.paper.payload,
                rung=9,
                provider=outcome.paper.provider,
                final_url=outcome.paper.final_url,
                http_status=outcome.paper.http_status,
                source_urls=tuple(outcome.paper.source_urls),
                identity_bound=True,
            )
            result.last_resort_pdf_path = outcome.pdf_path
            if winner := _accept(result, paper, validate):
                return winner

    # 10. Honest failure. The recipe turns this record into abstract_only/metadata_only.
    return result


# Hosts that only forward to a publisher. They have no account and no paywall of
# their own, so naming one as a Tier-4 login target sends the user somewhere that
# cannot possibly unlock anything.
_REDIRECTOR_HOSTS = frozenset(
    {"doi.org", "dx.doi.org", "hdl.handle.net", "handle.net", "n2t.net"}
)


def _is_redirector(url: str) -> bool:
    host = _hostname(url) or ""
    return host in _REDIRECTOR_HOSTS or host.removeprefix("www.") in _REDIRECTOR_HOSTS


# Bibliographic indexes. They describe a paper and never hold its body, so naming
# one as the publisher sent the reader to "omniread login nih.gov" for a paper
# that only bmj.com could unlock, and left the real publisher page unread.
_INDEX_HOSTS = frozenset(
    {
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "europepmc.org",
        "openalex.org",
        "api.openalex.org",
        "semanticscholar.org",
        "api.semanticscholar.org",
        "api.crossref.org",
        "search.crossref.org",
        "scholar.google.com",
    }
)


def _is_index(url: str) -> bool:
    host = (_hostname(url) or "").removeprefix("www.")
    return host in _INDEX_HOSTS


def _is_publisher_page(url: str) -> bool:
    return _is_http_url(url) and not _is_redirector(url) and not _is_index(url)


def _publisher_read_target(
    *,
    landing: HttpResponse | None,
    metadata: ScholarMetadata,
    original_url: str,
    doi: str | None,
) -> str | None:
    """Pick the publisher article page to read.

    Unlike ``_publisher_target``, which names a page a login could unlock, this may
    return the DOI itself: the generic ladder follows the redirect and reports the
    publisher it landed on, which is how a caller who only ever supplied a PubMed
    URL still gets the publisher's own article read.
    """

    if landing is not None and _is_publisher_page(landing.final_url):
        return landing.final_url
    # The DOI outranks any metadata landing URL: those are frequently repository
    # records (hal.science, an institutional archive), which rung 7 has already
    # mined and which are not the publisher. Resolving the DOI is what named
    # nejm.org instead of hal.science as the page that holds the paper.
    if doi:
        return f"https://doi.org/{quote(doi, safe='/')}"
    for location in metadata.oa_locations:
        if location.kind == "landing" and _is_publisher_page(location.url):
            return location.url
    if _is_publisher_page(original_url):
        return original_url
    return None


def _read_publisher_page(
    result: AcquisitionResult,
    ctx: RecipeContext,
    *,
    url: str,
    authenticated: bool,
) -> tuple[AcquiredPaper | None, str | None]:
    """Read one publisher page through the generic ladder and grade it as a body."""

    provider = (
        "tier-4-authenticated-render" if authenticated else "anonymous-publisher-render"
    )
    extra = {"auth_required": True} if authenticated else {}
    try:
        rendered = ctx.generic_read(
            url,
            budget=None,
            full=True,
            clock=ctx.clock,
            **extra,
        )
    except (OmniReadError, ValueError) as exc:
        result.attempts.append(AcquisitionAttempt(8, provider, url, False, str(exc)))
        return None, None
    final_url = rendered.provenance.final_url or url
    valid, detail = _classify_rendered(rendered)
    result.attempts.append(AcquisitionAttempt(8, provider, final_url, valid, detail))
    result.source_urls.extend(rendered.provenance.source_urls)
    observed = final_url if _is_publisher_page(final_url) else None
    if not valid:
        return None, observed
    paper = _rendered_paper(
        rendered,
        provider=provider,
        # An anonymous page is reached by following a locator rather than by a
        # work-specific mapping, so it must still prove its own title and author.
        identity_bound=authenticated,
    )
    return paper, observed


def _publisher_target(
    *,
    landing: HttpResponse | None,
    metadata: ScholarMetadata,
    original_url: str,
) -> str | None:
    """Pick a URL a Tier-4 login could actually unlock.

    A DOI that never resolved leaves only ``doi.org``, which forwards rather than
    authenticates, and a PubMed URL leaves only an index, which has no paywall of
    its own. Prefer the resolved landing page, then any publisher or repository
    landing URL the metadata sources named, and report no target at all rather
    than telling the caller to log in to a redirector or an index.
    """

    if landing is not None and _is_publisher_page(landing.final_url):
        return landing.final_url
    for location in metadata.oa_locations:
        if location.kind == "landing" and _is_publisher_page(location.url):
            return location.url
    if _is_publisher_page(original_url):
        return original_url
    return None


def is_public_candidate(
    url: str,
    *,
    resolve: HostResolver | None = None,
) -> bool:
    """Reject non-HTTP and non-public candidate hosts before fetching."""

    parsed = _safe_urlsplit(url)
    host = _hostname(url)
    if parsed is None or host is None:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    if _is_noncanonical_ip_literal(host):
        return False
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        resolver = resolve or resolve_host
        try:
            addresses = tuple(resolver(host))
        except (OSError, ValueError):
            return False
        if not addresses:
            return False
        try:
            parsed_addresses = tuple(ipaddress.ip_address(value) for value in addresses)
        except ValueError:
            return False
        return all(item.is_global for item in parsed_addresses)
    return literal.is_global


def _is_noncanonical_ip_literal(host: str) -> bool:
    """Reject URL-host spellings that network clients reinterpret as IP addresses."""

    if ":" in host:
        return False
    parts = host.split(".")

    def integer_component(value: str) -> bool:
        lowered = value.casefold()
        if lowered.startswith("0x"):
            return len(lowered) > 2 and all(
                character in "0123456789abcdef" for character in lowered[2:]
            )
        return value.isdigit()

    if not parts or not all(integer_component(part) for part in parts):
        return False
    if len(parts) != 4:
        return True
    for part in parts:
        lowered = part.casefold()
        if lowered.startswith("0x") or (len(part) > 1 and part.startswith("0")):
            return True
        if not part.isdigit() or int(part) > 255:
            return True
    return False


def resolve_host(host: str) -> tuple[str, ...]:
    """Resolve a host for the public-address guard."""

    records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


def citation_pdf_urls(html: str, base_url: str) -> list[str]:
    """Return de-duplicated citation PDF locators from one landing page."""

    urls: list[str] = []
    for node in HTMLParser(html).css("meta[name]"):
        if (node.attributes.get("name") or "").lower() != "citation_pdf_url":
            continue
        content = (node.attributes.get("content") or "").strip()
        if content:
            joined = _safe_urljoin(base_url, content)
            if joined is not None:
                urls.append(joined)
    return list(dict.fromkeys(urls))


def parse_unpaywall_locations(value: JsonValue) -> list[OALocation]:
    """Parse every Unpaywall OA PDF and landing location."""

    if not isinstance(value, dict):
        raise ValueError("Unpaywall returned an invalid metadata object")
    raw: list[JsonValue] = []
    best = value.get("best_oa_location")
    if isinstance(best, dict):
        raw.append(best)
    locations = value.get("oa_locations")
    if isinstance(locations, list):
        raw.extend(locations)
    parsed: list[OALocation] = []
    for location in raw:
        if not isinstance(location, dict):
            continue
        pdf = location.get("url_for_pdf")
        landing = location.get("url_for_landing_page") or location.get("url")
        if isinstance(pdf, str) and pdf:
            parsed.append(OALocation(pdf, "pdf", "unpaywall"))
        if isinstance(landing, str) and landing:
            parsed.append(OALocation(landing, "landing", "unpaywall"))
    return _deduplicate_locations(parsed)


def parse_biorxiv_jats(value: JsonValue) -> str | None:
    """Return bioRxiv/medRxiv's directly advertised JATS URL."""

    if not isinstance(value, dict):
        return None
    collection = value.get("collection")
    if not isinstance(collection, list):
        return None
    for item in collection:
        if isinstance(item, dict) and isinstance(item.get("jatsxml"), str):
            return str(item["jatsxml"])
    return None


def _attempt_candidate(
    result: AcquisitionResult,
    ctx: RecipeContext,
    *,
    rung: int,
    provider: str,
    url: str,
    expected: Literal["jats", "html", "pdf"],
    resolve: HostResolver,
    identity_bound: bool,
    minimum_visible_chars: int | None = None,
) -> AcquiredPaper | None:
    response = _fetch_response(
        result,
        ctx,
        rung=rung,
        provider=provider,
        url=url,
        resolve=resolve,
        record_success=False,
    )
    if response is None:
        return None
    valid, detail = _classify(
        response,
        expected,
        minimum_visible_chars=minimum_visible_chars,
    )
    result.attempts.append(
        AcquisitionAttempt(rung, provider, url, valid, detail)
    )
    if not valid:
        return None
    return AcquiredPaper(
        kind=expected,
        payload=response.body,
        rung=rung,
        provider=provider,
        final_url=response.final_url,
        http_status=response.status,
        source_urls=tuple(dict.fromkeys((url, response.final_url))),
        identity_bound=identity_bound,
    )


def _fetch_response(
    result: AcquisitionResult,
    ctx: RecipeContext,
    *,
    rung: int,
    provider: str,
    url: str,
    resolve: HostResolver,
    record_success: bool = True,
) -> HttpResponse | None:
    current_url = url
    visited = [url]
    for redirect_count in range(_MAX_REDIRECT_HOPS + 1):
        if not is_public_candidate(current_url, resolve=resolve):
            result.attempts.append(
                AcquisitionAttempt(
                    rung,
                    provider,
                    current_url,
                    False,
                    "Rejected before fetch because the candidate host is not public",
                )
            )
            return None
        try:
            response = get_with_retry(
                ctx.http,
                current_url,
                allow_redirects=False,
                max_bytes=_MAX_CANDIDATE_BYTES,
            )
        except RetrievalError as exc:
            result.attempts.append(
                AcquisitionAttempt(rung, provider, current_url, False, str(exc))
            )
            return None
        if 300 <= response.status < 400:
            location = response.headers.get("location")
            if not location:
                result.attempts.append(
                    AcquisitionAttempt(
                        rung,
                        provider,
                        current_url,
                        False,
                        "Redirect response exposed no Location header",
                    )
                )
                return None
            if redirect_count == _MAX_REDIRECT_HOPS:
                result.attempts.append(
                    AcquisitionAttempt(
                        rung,
                        provider,
                        current_url,
                        False,
                        f"Candidate exceeded the {_MAX_REDIRECT_HOPS}-redirect limit",
                    )
                )
                return None
            joined = _safe_urljoin(response.final_url, location)
            if joined is None:
                result.attempts.append(
                    AcquisitionAttempt(
                        rung,
                        provider,
                        current_url,
                        False,
                        "Redirect response exposed a malformed Location URL",
                    )
                )
                return None
            current_url = joined
            visited.append(current_url)
            continue
        if not is_public_candidate(response.final_url, resolve=resolve):
            result.attempts.append(
                AcquisitionAttempt(
                    rung,
                    provider,
                    response.final_url,
                    False,
                    "Rejected because the fetched candidate resolved to a non-public host",
                )
            )
            return None
        break
    else:  # pragma: no cover - the bounded loop always returns or breaks
        return None
    result.source_urls.extend((*visited, response.final_url))
    if record_success:
        result.attempts.append(
            AcquisitionAttempt(rung, provider, url, True, "Fetched landing representation")
        )
    return response


def _classify(
    response: HttpResponse,
    expected: Literal["jats", "html", "pdf"],
    *,
    minimum_visible_chars: int | None = None,
) -> tuple[bool, str]:
    if expected == "pdf":
        if response.body.startswith(b"%PDF"):
            return True, "Response begins with PDF magic bytes"
        return False, "Advertised PDF did not begin with PDF magic bytes"
    if expected == "jats":
        try:
            root = parse_jats_xml(response.body)
        except (ValueError, SyntaxError) as exc:
            return False, f"Advertised JATS response is unsafe or invalid XML: {exc}"
        body = next((node for node in root.iter() if _local_name(node.tag) == "body"), None)
        if body is None or not " ".join(body.itertext()).strip():
            return False, "JATS response contains no non-empty body element"
        return True, "JATS response contains a non-empty body element"
    tree = HTMLParser(response.text)
    body = tree.body
    if body is None:
        return False, "HTML response contains no body element"
    visible = " ".join(body.text().split())
    minimum = minimum_visible_chars or 1_000
    if len(visible) < minimum:
        return False, f"HTML body is too short to be scholarly full text ({len(visible)} chars)"
    return (
        True,
        f"HTML contains {len(visible)} visible characters and remains subject to "
        "artifact-level scholarly verification",
    )


def _classify_rendered(result: ReadResult) -> tuple[bool, str]:
    """Gate a rendered publisher representation for artifact-level verification.

    Rung 8 reads the publisher page anonymously as well as behind a stored login,
    so this wording stays neutral: reporting an anonymous read as "authenticated"
    told the reader a login had been used when none had.
    """

    content = result.content.strip()
    if not content:
        return False, "The publisher page returned no usable content"
    if len(content) < 1_000:
        return (
            False,
            f"The publisher page is too short for scholarly body verification "
            f"({len(content)} chars)",
        )
    return (
        True,
        f"The publisher page has {len(content)} characters and remains "
        "subject to artifact-level scholarly verification",
    )


def _fetch_unpaywall(
    doi: str,
    result: AcquisitionResult,
    ctx: RecipeContext,
) -> list[OALocation]:
    contact = os.environ.get("OMNIREAD_CONTACT_EMAIL", "").strip()
    if not contact:
        result.attempts.append(
            AcquisitionAttempt(
                5,
                "unpaywall",
                "https://api.unpaywall.org/v2/",
                False,
                "OMNIREAD_CONTACT_EMAIL is not configured",
            )
        )
        return []
    url = (
        f"https://api.unpaywall.org/v2/{quote(doi, safe='')}"
        f"?email={quote(contact, safe='@')}"
    )
    try:
        response = get_with_retry(ctx.http, url)
        locations = parse_unpaywall_locations(response.json())
    except (RetrievalError, TypeError, ValueError) as exc:
        result.attempts.append(
            AcquisitionAttempt(5, "unpaywall", url, False, str(exc))
        )
        return []
    result.source_urls.extend((url, response.final_url))
    result.attempts.append(
        AcquisitionAttempt(
            5,
            "unpaywall",
            url,
            bool(locations),
            f"Unpaywall exposed {len(locations)} candidate locations",
        )
    )
    return locations


def _published_preprints(
    doi: str,
    result: AcquisitionResult,
    ctx: RecipeContext,
) -> list[tuple[str, str]]:
    """Resolve a publisher DOI to bioRxiv/medRxiv DOI mappings."""

    linked: list[tuple[str, str]] = []
    failed_lookup = False
    for server in ("biorxiv", "medrxiv"):
        url = f"https://api.biorxiv.org/pubs/{server}/{quote(doi, safe='/')}"
        try:
            response = get_with_retry(ctx.http, url)
            preprint_doi = _parse_preprint_doi(response.json())
        except (RetrievalError, TypeError, ValueError) as exc:
            failed_lookup = True
            result.attempts.append(
                AcquisitionAttempt(
                    2,
                    f"{server}-pubs",
                    url,
                    False,
                    f"Mapping lookup did not succeed: {exc}",
                )
            )
            continue
        result.source_urls.extend((url, response.final_url))
        if preprint_doi:
            linked.append((server, preprint_doi))
            detail = f"Successful mapping lookup linked preprint DOI {preprint_doi}"
        else:
            detail = "Successful mapping lookup exposed no linked preprint"
        result.attempts.append(
            AcquisitionAttempt(
                2,
                f"{server}-pubs",
                url,
                bool(preprint_doi),
                detail,
            )
        )
    if not linked and not failed_lookup:
        result.attempts.append(
            AcquisitionAttempt(
                2,
                "preprint-cross-reference",
                f"https://api.biorxiv.org/pubs/biorxiv/{quote(doi, safe='/')}",
                False,
                "Both mapping requests succeeded; no preprint linked to this DOI",
            )
        )
    return list(dict.fromkeys(linked))


def _parse_preprint_doi(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    collection = value.get("collection")
    if not isinstance(collection, list):
        return None
    for item in collection:
        if not isinstance(item, dict):
            continue
        candidate = item.get("preprint_doi")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return None


def _biorxiv_jats_url(
    api_url: str,
    result: AcquisitionResult,
    ctx: RecipeContext,
) -> str | None:
    try:
        response = get_with_retry(ctx.http, api_url)
        jats_url = parse_biorxiv_jats(response.json())
    except (RetrievalError, TypeError, ValueError) as exc:
        result.attempts.append(
            AcquisitionAttempt(2, "biorxiv-api", api_url, False, str(exc))
        )
        return None
    result.source_urls.extend((api_url, response.final_url))
    result.attempts.append(
        AcquisitionAttempt(
            2,
            "biorxiv-api",
            api_url,
            jats_url is not None,
            "API exposed a JATS URL" if jats_url else "API exposed no JATS URL",
        )
    )
    return jats_url


def _rendered_paper(
    result: ReadResult,
    *,
    provider: str,
    identity_bound: bool,
) -> AcquiredPaper | None:
    if not result.content.strip():
        return None
    return AcquiredPaper(
        kind="markdown",
        payload=result.content.encode("utf-8"),
        rung=8,
        provider=provider,
        final_url=result.provenance.final_url,
        http_status=result.provenance.http_status,
        source_urls=tuple(result.provenance.source_urls),
        identity_bound=identity_bound,
        tier=result.provenance.tier,
    )


def _won(result: AcquisitionResult, paper: AcquiredPaper) -> AcquisitionResult:
    result.paper = paper
    result.source_urls.extend(paper.source_urls)
    result.source_urls = list(dict.fromkeys(result.source_urls))
    return result


def _accept(
    result: AcquisitionResult,
    paper: AcquiredPaper,
    validate: CandidateValidator | None,
) -> AcquisitionResult | None:
    if validate is not None:
        accepted, detail = validate(paper)
        if not accepted:
            result.attempts.append(
                AcquisitionAttempt(
                    paper.rung,
                    f"{paper.provider}-extraction",
                    paper.final_url,
                    False,
                    detail,
                )
            )
            return None
    return _won(result, paper)


def _locations(
    values: Iterable[OALocation],
    *,
    provider_prefix: str | None = None,
    kind: str | None = None,
) -> list[OALocation]:
    return [
        item
        for item in values
        if (provider_prefix is None or item.provider.startswith(provider_prefix))
        and (kind is None or item.kind == kind)
    ]


def _deduplicate_locations(values: Iterable[OALocation]) -> list[OALocation]:
    seen: set[str] = set()
    result: list[OALocation] = []
    for item in values:
        if item.url not in seen:
            result.append(item)
            seen.add(item.url)
    return result


def _host_has(url: str, *domains: str) -> bool:
    host = _hostname(url) or ""
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _arxiv_from_doi(doi: str | None) -> str | None:
    if doi and doi.lower().startswith("10.48550/arxiv."):
        return doi[len("10.48550/arxiv.") :]
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_urlsplit(value: str) -> SplitResult | None:
    """Parse an untrusted fetched URL without letting parser errors escape."""

    try:
        return urlsplit(value)
    except (TypeError, ValueError):
        return None


def _hostname(value: str) -> str | None:
    parsed = _safe_urlsplit(value)
    if parsed is None:
        return None
    try:
        host = parsed.hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def _is_http_url(value: str) -> bool:
    parsed = _safe_urlsplit(value)
    return bool(
        parsed is not None
        and parsed.scheme in {"http", "https"}
        and _hostname(value) is not None
    )


def _safe_urljoin(base_url: str, candidate: str) -> str | None:
    try:
        joined = urljoin(base_url, candidate)
    except (TypeError, ValueError):
        return None
    return joined if _safe_urlsplit(joined) is not None else None
