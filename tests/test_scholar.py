from __future__ import annotations

from datetime import datetime, timezone
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from omniread.auth import ProfileStore
from omniread.extract import extract_page
from omniread.recipes.academic import AcademicRecipe
from omniread.recipes.base import (
    HttpResponse,
    ReadOptions,
    RecipeContext,
    RecipeHttpClient,
)
from omniread.recipes import RecipeRegistry
from omniread.scholar import locate as locate_module
from omniread.scholar import http as scholar_http_module
from omniread.scholar import extract as extract_module
from omniread.scholar.extract import (
    ScholarExtraction,
    extract_abstract_from_html,
    extract_html,
    extract_jats,
    extract_pdf,
    has_text_layer,
)
from omniread.scholar.ids import (
    SCHOLARLY_HOSTS,
    ScholarlyIdentifier,
    extract_meta_doi,
    is_scholarly_input,
    normalize_identifier,
)
from omniread.scholar.http import get_with_retry
from omniread.scholar.identity import verify_identity
from omniread.scholar.locate import (
    AcquiredPaper,
    acquire_full_text,
    citation_pdf_urls,
    is_public_candidate,
    parse_biorxiv_jats,
    parse_unpaywall_locations,
)
from omniread.scholar.metadata import (
    OALocation,
    ScholarMetadata,
    collect_metadata,
    parse_europe_pmc,
    parse_openalex,
)
from omniread.scholar.verify import verify_scholarly
from omniread.tokens import build_outline
from omniread.types import (
    Completeness,
    Evidence,
    ExtractionError,
    Provenance,
    ReadResult,
    RetrievalError,
)

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
PUBLIC_RESOLVER = lambda host: ("93.184.216.34",)


class _StubHttp:
    def __init__(self, routes: list[tuple[str, HttpResponse | Exception]]) -> None:
        self.routes = routes
        self.calls: list[str] = []
        self.call_options: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        headers=None,
        allow_redirects: bool = True,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        self.call_options.append(
            {
                "headers": headers,
                "allow_redirects": allow_redirects,
                "max_bytes": max_bytes,
            }
        )
        for marker, outcome in self.routes:
            if marker in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise RetrievalError(f"No fixture route for {url}")


def _response(
    fixture: str | bytes,
    *,
    final_url: str,
    status: int = 200,
    content_type: str = "text/html",
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    body = fixture if isinstance(fixture, bytes) else (FIXTURES / fixture).read_bytes()
    return HttpResponse(
        body=body,
        status=status,
        final_url=final_url,
        headers={"content-type": content_type, **(headers or {})},
    )


def _minimal_text_pdf(lines: list[str]) -> bytes:
    """Build a one-page born-digital PDF without adding a test dependency."""

    escaped = [
        line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        for line in lines
    ]
    commands = ["BT /F1 11 Tf 72 740 Td 14 TL"]
    for index, line in enumerate(escaped):
        commands.append(f"({line}) Tj" if index == 0 else f"T* ({line}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(payload)


def _jats_paper(*, title: str, surname: str) -> bytes:
    body = " ".join(f"observation{index}" for index in range(1_100))
    references = "".join(
        f"<ref id='r{index}'><mixed-citation>Reference {index}</mixed-citation></ref>"
        for index in range(1, 21)
    )
    return (
        "<article><front><article-meta>"
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"<contrib-group><contrib><name><surname>{surname}</surname></name></contrib>"
        "</contrib-group></article-meta></front><body>"
        f"<sec><title>Methods</title><p>{body}</p></sec>"
        f"<sec><title>Results</title><p>{body}</p></sec>"
        f"</body><back><ref-list>{references}</ref-list></back></article>"
    ).encode()


def _unreachable_publisher_page(*args, **kwargs) -> ReadResult:
    """Model a publisher page rung 8 cannot read, and keep the Tier-4 guard exact.

    Rung 8 now reads the publisher's own article page anonymously before any login
    is considered, so a stub that refuses every call would assert against the
    contract rather than the behaviour. An authenticated call without a stored
    profile is still a bug, so that one keeps failing loudly.
    """

    if kwargs.get("auth_required"):
        raise AssertionError("authenticated render was not expected")
    raise RetrievalError("the publisher page is unreachable in this test")


def _context(tmp_path: Path, http: _StubHttp, *, generic_read=None) -> RecipeContext:
    return RecipeContext(
        clock=lambda: NOW,
        generic_read=generic_read or _unreachable_publisher_page,
        http=http,  # type: ignore[arg-type]
        profiles=ProfileStore(tmp_path / "profiles"),
        render=None,
    )


@pytest.mark.parametrize(
    "value",
    [
        "10.1038/nature14539",
        "https://doi.org/10.1038/nature14539",
        "https://arxiv.org/abs/1706.03762v7",
        "https://arxiv.org/pdf/1706.03762.pdf",
        "https://pubmed.ncbi.nlm.nih.gov/26017442/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1182327/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1182327/",
    ],
)
def test_scholarly_identifiers_normalize_without_network(value: str) -> None:
    identifier = normalize_identifier(value)

    assert identifier.kind in {"doi", "arxiv", "pmid", "pmcid"}
    assert identifier.value


def test_publisher_url_recovers_doi_from_standard_meta() -> None:
    html = (FIXTURES / "scholar_nature_paywall.html").read_text()
    url = "https://www.nature.com/articles/nature14539"

    assert extract_meta_doi(html) == "10.1038/nature14539"
    assert normalize_identifier(url, page_html=html) == ScholarlyIdentifier(
        "doi", "10.1038/nature14539"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://doi.org/10.1038/nature14539",
        "https://dx.doi.org/10.1038/nature14539",
        "https://DX.DOI.ORG./10.1038/nature14539",
        "https://hdl.handle.net/10.1038/nature14539",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1182327/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1182327/",
        "https://www.pmc.ncbi.nlm.nih.gov./articles/PMC1182327/",
    ],
)
def test_scholarly_alias_reproductions_route_to_academic_recipe(url: str) -> None:
    registry = RecipeRegistry.discover(local_paths=(), include_entry_points=False)

    assert is_scholarly_input(url)
    assert AcademicRecipe.match(url)
    assert registry.find(url).name == "academic"


def _scholarly_article_examples() -> dict[str, str]:
    examples = {
        "academic.oup.com": "/journal/article/1/1/1/1",
        "arxiv.org": "/abs/1706.03762",
        "biorxiv.org": "/content/10.1101/2024.01.01.123456v1",
        "bmj.com": "/content/372/bmj.n71",
        "cell.com": "/cell/fulltext/S0092-8674(11)00127-9",
        "chemrxiv.org": "/engage/chemrxiv/article-details/fixture",
        "dl.acm.org": "/doi/10.1145/fixture",
        "doi.org": "/10.1038/nature14539",
        "elifesciences.org": "/articles/00001",
        "europepmc.org": "/articles/PMC1182327",
        "frontiersin.org": "/journals/test/articles/10.3389/fixture/full",
        "hal.science": "/hal-04206682",
        "hdl.handle.net": "/10.1038/nature14539",
        "ieeexplore.ieee.org": "/document/1234567",
        "jamanetwork.com": "/journals/jama/fullarticle/1234567",
        "journals.plos.org": "/plosone/article?id=10.1371/journal.pone.0000001",
        "jstor.org": "/stable/1914185",
        "link.springer.com": "/article/10.1007/s11263-015-0816-y",
        "linkinghub.elsevier.com": "/retrieve/pii/S0092867411001279",
        "mdpi.com": "/2072-4292/16/1/1",
        "medrxiv.org": "/content/10.1101/2024.01.01.123456v1",
        "nature.com": "/articles/nature14539",
        "ncbi.nlm.nih.gov": "/pmc/articles/PMC1182327/",
        "nejm.org": "/doi/full/10.1056/NEJMoa2034577",
        "onlinelibrary.wiley.com": "/doi/10.1111/fixture",
        "openalex.org": "/works/W123456789",
        "osf.io": "/preprints/psyarxiv/abcde",
        "pmc.ncbi.nlm.nih.gov": "/articles/PMC1182327/",
        "pnas.org": "/doi/10.1073/pnas.1234567890",
        "psyarxiv.com": "/abcde",
        "pubmed.ncbi.nlm.nih.gov": "/26017442/",
        "researchgate.net": "/publication/123456789_Paper",
        "sagepub.com": "/doi/10.1177/fixture",
        "science.org": "/doi/10.1126/science.fixture",
        "sciencedirect.com": "/science/article/pii/S0092867411001279",
        "semanticscholar.org": "/paper/title/abcdef",
        "ssrn.com": "/abstract=1234567",
        "tandfonline.com": "/doi/full/10.1080/fixture",
        "thelancet.com": "/journals/lancet/article/PIIS0140-6736(20)00000-0/fulltext",
        "zenodo.org": "/records/1234567",
    }
    return examples


def test_scholarly_host_catalog_has_an_article_shaped_reproduction_per_host() -> None:
    assert set(_scholarly_article_examples()) == set(SCHOLARLY_HOSTS)


@pytest.mark.parametrize(
    ("host", "path"),
    sorted(_scholarly_article_examples().items()),
)
def test_academic_recipe_owns_article_shaped_scholarly_urls(
    host: str,
    path: str,
) -> None:
    assert AcademicRecipe.match(f"https://{host}{path}")
    assert AcademicRecipe.match(f"https://www.{host}.{path}")


@pytest.mark.parametrize(
    "value",
    [
        "https://www.nature.com/",
        "https://www.nature.com/news/example",
        "https://www.ncbi.nlm.nih.gov/books/NBK555591/",
        "https://blast.ncbi.nlm.nih.gov/Blast.cgi",
        "https://www.science.org/careers/example",
        "https://www.researchgate.net/profile/example",
        "https://www.researchgate.net/post/X",
        "https://zenodo.org/communities/oa",
        "https://osf.io/dashboard",
        "https://osf.io/myprojects",
        "https://www.science.org/content/article/news-slug",
        "https://www.nature.com/nature",
        "https://www.nature.com/subjects/machine-learning",
        "https://www.nature.com/collections/x",
        "https://www.sciencedirect.com/journal/the-lancet",
        "https://journals.plos.org/plosone/",
        "https://arxiv.org/list/cs.LG/recent",
        "https://www.biorxiv.org/search/deep+learning",
        "https://www.jstor.org/action/doBasicSearch?Query=x",
    ],
)
def test_academic_recipe_declines_non_article_inputs(value: str) -> None:
    assert not AcademicRecipe.match(value)


def test_bare_numeric_pmid_remains_valid_only_as_explicit_identifier() -> None:
    """A run of digits is a PMID when the caller says so, never by discovery.

    This previously asserted `AcademicRecipe.match("12345")`, which contradicts the
    test's own name and encodes the over-claim an adversarial review flagged: any
    4-to-10 digit string routed into the paper ladder. Normalizing an explicit
    identifier is a different act from claiming an arbitrary input.
    """

    assert normalize_identifier("12345") == ScholarlyIdentifier("pmid", "12345")
    assert not AcademicRecipe.match("12345")


def test_metadata_parsers_keep_independent_reference_counts_and_oa_locations() -> None:
    openalex = parse_openalex(
        json.loads((FIXTURES / "scholar_openalex_oa.json").read_text())
    )
    epmc = parse_europe_pmc(
        json.loads((FIXTURES / "scholar_epmc_search.json").read_text()),
        expected=ScholarlyIdentifier("doi", "10.1056/nejmoa2034577"),
    )
    unpaywall = parse_unpaywall_locations(
        json.loads((FIXTURES / "scholar_unpaywall.json").read_text())
    )
    biorxiv = parse_biorxiv_jats(
        json.loads((FIXTURES / "scholar_biorxiv.json").read_text())
    )

    assert openalex.abstract and openalex.abstract.startswith("There is increasing concern")
    assert openalex.openalex_reference_count == 40
    assert any(item.kind == "pdf" for item in openalex.oa_locations)
    assert epmc.pmcid == "PMC7745181"
    assert epmc.pmid == "33301246"
    assert any(item.kind == "landing" for item in unpaywall)
    assert biorxiv and biorxiv.endswith(".source.xml")


def test_openalex_arxiv_id_ignores_referenced_and_related_works() -> None:
    metadata = parse_openalex(
        {
            "title": "Deep learning",
            "ids": {},
            "primary_location": {
                "landing_page_url": "https://doi.org/10.1038/nature14539"
            },
            "locations": [],
            "referenced_works": ["https://arxiv.org/abs/1409.4842"],
            "related_works": ["https://arxiv.org/abs/1512.03385"],
        }
    )

    assert metadata.arxiv_id is None
    assert all("arxiv.org" not in item.url for item in metadata.oa_locations)


def test_openalex_arxiv_id_accepts_only_this_works_own_id() -> None:
    metadata = parse_openalex(
        {
            "title": "Densely connected convolutional networks",
            "ids": {"arxiv": "https://arxiv.org/abs/1608.06993"},
            "referenced_works": ["https://arxiv.org/abs/1409.4842"],
        }
    )

    assert metadata.arxiv_id == "1608.06993"


def test_identity_gate_rejects_wrong_paper_from_unbound_locator() -> None:
    wrong = (
        "<html><body><h1>Going Deeper with Convolutions</h1>"
        "<p>Christian Szegedy, Wei Liu, Yangqing Jia</p></body></html>"
    ).encode()

    accepted, detail = verify_identity(
        wrong,
        "html",
        ScholarMetadata(
            title="Deep learning",
            authors=["LeCun", "Bengio", "Hinton"],
        ),
        bound=False,
    )

    assert not accepted
    assert "author" in detail.lower()


def test_identity_gate_requires_title_and_author_for_unbound_locator() -> None:
    right = (
        "<html><body><h1>Deep learning</h1>"
        "<p>Yann LeCun, Yoshua Bengio, Geoffrey Hinton</p></body></html>"
    ).encode()

    accepted, detail = verify_identity(
        right,
        "html",
        ScholarMetadata(
            title="Deep learning",
            authors=["LeCun", "Bengio", "Hinton"],
        ),
        bound=False,
    )

    assert accepted, detail


def test_identity_gate_accepts_bound_retitled_preprint_with_matching_author() -> None:
    retitled = _jats_paper(
        title=(
            "Language models of protein sequences at the scale of evolution enable "
            "accurate structure prediction"
        ),
        surname="Lin",
    )

    accepted, detail = verify_identity(
        retitled,
        "jats",
        ScholarMetadata(
            title=(
                "Evolutionary-scale prediction of atomic-level protein structure "
                "with a language model"
            ),
            authors=["Lin", "Akin"],
        ),
        bound=True,
    )

    assert accepted, detail


def test_retrying_scholarly_fetch_backs_off_on_429_and_5xx() -> None:
    responses = [
        HttpResponse(b"", 429, "https://api.example/paper", {}),
        HttpResponse(b"", 503, "https://api.example/paper", {}),
        HttpResponse(b"ok", 200, "https://api.example/paper", {}),
    ]
    calls: list[str] = []
    sleeps: list[float] = []

    class SequenceHttp:
        def get(self, url: str, **kwargs) -> HttpResponse:
            calls.append(url)
            return responses.pop(0)

    response = get_with_retry(
        SequenceHttp(),  # type: ignore[arg-type]
        "https://api.example/paper",
        sleeper=sleeps.append,
    )

    assert response.body == b"ok"
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_metadata_sources_fail_independently_without_raising() -> None:
    http = _StubHttp(
        [
            (
                "api.openalex.org",
                _response(
                    "scholar_openalex_paywalled.json",
                    final_url="https://api.openalex.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            (
                "api.crossref.org",
                _response(
                    "scholar_crossref_paywalled.json",
                    final_url="https://api.crossref.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            ("europepmc", RetrievalError("Europe PMC unavailable")),
        ]
    )

    metadata = collect_metadata(
        ScholarlyIdentifier("doi", "10.1038/nature14539"),
        http,  # type: ignore[arg-type]
    )

    assert metadata.title == "Deep learning"
    assert metadata.crossref_reference_count == 103
    assert metadata.openalex_reference_count == 53
    assert (metadata.first_page, metadata.last_page) == ("436", "444")
    assert metadata.pmid == "26017442"
    assert metadata.failures["europe_pmc"] == "Europe PMC unavailable"


def test_contact_email_is_stripped_from_every_structured_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contact = "reader@example.test"
    monkeypatch.setenv("OMNIREAD_CONTACT_EMAIL", contact)
    unpaywall_url = (
        "https://api.unpaywall.org/v2/10.1234%2Fcontact"
        f"?email={contact}"
    )
    http = _StubHttp(
        [
            (
                "api.unpaywall.org",
                _response(
                    json.dumps({"oa_locations": []}).encode(),
                    final_url=unpaywall_url,
                    content_type="application/json",
                ),
            )
        ]
    )
    metadata = ScholarMetadata(
        doi="10.1234/contact",
        oa_locations=[
            OALocation(
                f"https://repository.example/paper?mailto={contact}&version=2",
                "landing",
                "openalex",
            )
        ],
    )
    acquisition = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/contact"),
        metadata=metadata,
        original_url="https://doi.org/10.1234/contact",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )
    exposed = json.dumps(
        {
            "metadata": metadata.as_json(),
            "attempts": [attempt.as_json() for attempt in acquisition.attempts],
        }
    )

    assert contact not in exposed
    assert "mailto=" not in exposed
    assert "email=" not in exposed
    assert "version=2" in exposed


def test_pmc_input_resolves_doi_then_collects_all_three_manifests() -> None:
    http = _StubHttp(
        [
            (
                "europepmc",
                _response(
                    "scholar_epmc_search.json",
                    final_url="https://www.ebi.ac.uk/europepmc/search",
                    content_type="application/json",
                ),
            ),
            (
                "api.openalex.org",
                _response(
                    json.dumps(
                        {
                            "doi": "https://doi.org/10.1056/nejmoa2034577",
                            "title": "Vaccine trial",
                            "referenced_works_count": 8,
                        }
                    ).encode(),
                    final_url="https://api.openalex.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            (
                "api.crossref.org",
                _response(
                    json.dumps(
                        {
                            "message": {
                                "DOI": "10.1056/nejmoa2034577",
                                "title": ["Vaccine trial"],
                                "reference-count": 13,
                                "page": "2603-2615",
                            }
                        }
                    ).encode(),
                    final_url="https://api.crossref.org/works/fixture",
                    content_type="application/json",
                ),
            ),
        ]
    )

    metadata = collect_metadata(
        ScholarlyIdentifier("pmcid", "PMC7745181"),
        http,  # type: ignore[arg-type]
    )

    assert metadata.doi == "10.1056/nejmoa2034577"
    assert metadata.openalex_reference_count == 8
    assert metadata.crossref_reference_count == 13
    assert ["europepmc" in http.calls[0], "openalex" in http.calls[1], "crossref" in http.calls[2]] == [
        True,
        True,
        True,
    ]


def test_jats_fixture_yields_full_text_with_references() -> None:
    payload = (FIXTURES / "scholar_epmc_jats.xml").read_bytes()
    extraction = extract_jats(payload)
    paper = AcquiredPaper(
        "jats",
        payload,
        1,
        "europe-pmc-jats",
        "https://europepmc.org/articles/PMC1182327",
        200,
        ("https://europepmc.org/articles/PMC1182327",),
    )

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(),
        acquisition=paper,
    )

    assert "## Modeling the Framework for False Positive Findings" in extraction.markdown
    assert "## References" in extraction.markdown
    assert extraction.reference_count == 37
    assert verdict.content_level == "full_text_with_references"
    assert verdict.completeness.status == "complete"


def test_arxiv_html_fixture_yields_full_text() -> None:
    url = "https://arxiv.org/html/1706.03762"
    payload = (FIXTURES / "scholar_arxiv.html").read_bytes()
    extraction = extract_html(payload.decode(), url)
    paper = AcquiredPaper(
        "html", payload, 3, "arxiv-html", url, 200, (url,)
    )

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(arxiv_id="1706.03762"),
        acquisition=paper,
    )

    assert "## 1 Introduction" in extraction.markdown
    assert "## 7 Conclusion" in extraction.markdown
    assert extraction.reference_count == 40
    assert verdict.content_level == "full_text"
    assert verdict.completeness.status == "complete"


def test_pypdf_is_default_for_born_digital_papers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_text_pdf(
        [
            "Introduction",
            "This paper contains a real text layer and readable body prose.",
            "Methods",
            "The method is independently described in this second section.",
            "References",
            "1. Example reference",
        ]
    )

    def forbidden_docling(_: bytes) -> str:
        raise AssertionError("Docling must not run for a PDF with a text layer")

    monkeypatch.setattr(extract_module, "_docling_markdown", forbidden_docling)
    extraction = extract_pdf(payload)

    assert has_text_layer(payload) is True
    assert extraction.section_titles[:2] == ("Introduction", "Methods")
    assert extraction.reference_count == 1
    assert "real text layer" in extraction.markdown


def test_docling_conversion_errors_become_candidate_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenConverter:
        def convert(self, path: str) -> object:
            raise RuntimeError(f"malformed PDF at {path}")

    monkeypatch.setattr(extract_module, "has_text_layer", lambda payload: False)
    monkeypatch.setattr(
        extract_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(DocumentConverter=BrokenConverter),
    )

    with pytest.raises(ExtractionError, match="Docling could not convert"):
        extract_pdf(b"%PDF-malformed")


def test_jats_entity_expansion_is_rejected_before_parsing() -> None:
    payload = b"""<?xml version="1.0"?>
    <!DOCTYPE article [
      <!ENTITY a "1234567890">
      <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
    ]>
    <article><body><sec><title>Methods</title><p>&b;</p></sec></body></article>
    """
    response = HttpResponse(
        payload,
        200,
        "https://repository.example/paper.xml",
        {"content-type": "application/xml"},
    )

    with pytest.raises(ExtractionError, match="entity declarations"):
        extract_jats(payload)
    valid, detail = locate_module._classify(response, "jats")
    assert valid is False
    assert "entity declarations" in detail


def test_pdf_masquerade_is_rejected_as_html_and_does_not_win(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    url = "https://www.nature.com/articles/nature14539.pdf"
    http = _StubHttp(
        [
            (
                url,
                _response("scholar_pdf_masquerade.html", final_url=url),
            )
        ]
    )
    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1038/nature14539"),
        metadata=ScholarMetadata(
            doi="10.1038/nature14539",
            oa_locations=[OALocation(url, "pdf", "openalex")],
        ),
        original_url="https://www.nature.com/articles/nature14539",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    assert any(
        attempt.url == url
        and not attempt.succeeded
        and "magic bytes" in attempt.detail
        for attempt in result.attempts
    )


def test_published_doi_cross_reference_acquires_retitled_preprint(
    tmp_path: Path,
) -> None:
    published_doi = "10.1126/science.ade2574"
    preprint_doi = "10.1101/2022.07.20.500902"
    jats_url = "https://www.biorxiv.org/content/fixture.source.xml"
    retitled_jats = _jats_paper(
        title=(
            "Language models of protein sequences at the scale of evolution enable "
            "accurate structure prediction"
        ),
        surname="Lin",
    )
    http = _StubHttp(
        [
            (
                "/pubs/biorxiv/",
                _response(
                    json.dumps(
                        {"collection": [{"preprint_doi": preprint_doi}]}
                    ).encode(),
                    final_url=f"https://api.biorxiv.org/pubs/biorxiv/{published_doi}",
                    content_type="application/json",
                ),
            ),
            (
                "/pubs/medrxiv/",
                _response(
                    json.dumps({"collection": []}).encode(),
                    final_url=f"https://api.biorxiv.org/pubs/medrxiv/{published_doi}",
                    content_type="application/json",
                ),
            ),
            (
                f"/details/biorxiv/{preprint_doi}",
                _response(
                    json.dumps({"collection": [{"jatsxml": jats_url}]}).encode(),
                    final_url=f"https://api.biorxiv.org/details/biorxiv/{preprint_doi}",
                    content_type="application/json",
                ),
            ),
            (
                jats_url,
                _response(
                    retitled_jats,
                    final_url=jats_url,
                    content_type="application/xml",
                ),
            ),
        ]
    )
    metadata = ScholarMetadata(
        doi=published_doi,
        title=(
            "Evolutionary-scale prediction of atomic-level protein structure "
            "with a language model"
        ),
        authors=["Lin", "Akin"],
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", published_doi),
        metadata=metadata,
        original_url=f"https://doi.org/{published_doi}",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
        validate=lambda paper: verify_identity(
            paper.payload,
            paper.kind,
            metadata,
            bound=paper.identity_bound,
        ),
    )

    assert result.paper is not None
    assert result.paper.provider == "biorxiv-jats"
    assert result.paper.identity_bound is True
    assert any(preprint_doi in call for call in http.calls)


def test_europe_pmc_fetches_resolved_pmcid_even_when_not_open_access(
    tmp_path: Path,
) -> None:
    doi = "10.1001/jamapsychiatry.fixture"
    pmcid = "PMC6137521"
    metadata = parse_europe_pmc(
        {
            "resultList": {
                "result": [
                    {
                        "doi": doi,
                        "pmcid": pmcid,
                        "title": "A funder-deposited manuscript",
                        "isOpenAccess": "N",
                    }
                ]
            }
        },
        expected=ScholarlyIdentifier("doi", doi),
    )
    jats_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        f"{pmcid}/fullTextXML"
    )
    http = _StubHttp(
        [
            (
                "fullTextXML",
                _response(
                    _jats_paper(title=metadata.title or "Paper", surname="Lewis"),
                    final_url=jats_url,
                    content_type="application/xml",
                ),
            )
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", doi),
        metadata=metadata,
        original_url=f"https://doi.org/{doi}",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert metadata.pmcid == pmcid
    assert result.paper is not None
    assert result.paper.provider == "europe-pmc-jats"
    assert "fullTextXML" in http.calls[0]


def test_pmc_direct_falls_back_from_xml_and_pdf_to_substantial_article_page(
    tmp_path: Path,
) -> None:
    pmcid = "PMC6137521"
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    body = " ".join(f"finding{index}" for index in range(2_500))
    article = (
        "<html><body><h1>Population mental health outcomes</h1>"
        "<p>A. Lewis and colleagues</p><h2>Methods</h2>"
        f"<p>{body}</p><h2>Results</h2><p>{body}</p></body></html>"
    ).encode()
    http = _StubHttp(
        [
            (
                "fullTextXML",
                _response(b"", final_url="https://europepmc.example/xml", status=404),
            ),
            (
                f"/{pmcid}/pdf/",
                _response(b"", final_url=f"{article_url}pdf/", status=404),
            ),
            (
                article_url,
                _response(article, final_url=article_url),
            ),
        ]
    )
    metadata = ScholarMetadata(
        doi="10.1001/jamapsychiatry.fixture",
        pmcid=pmcid,
        title="Population mental health outcomes",
        authors=["Lewis"],
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("pmcid", pmcid),
        metadata=metadata,
        original_url=article_url,
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
        validate=lambda paper: verify_identity(
            paper.payload,
            paper.kind,
            metadata,
            bound=paper.identity_bound,
        ),
    )

    assert result.paper is not None
    assert result.paper.provider == "pmc-direct-html"
    assert result.paper.kind == "html"
    assert http.calls[:3] == [
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6137521/fullTextXML",
        f"{article_url}pdf/",
        article_url,
    ]


def test_pmc_direct_rejects_a_short_landing_shell(tmp_path: Path) -> None:
    pmcid = "PMC6137521"
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    shell = (
        "<html><body><h1>Population mental health outcomes</h1>"
        "<p>A. Lewis</p><p>Navigation and citation tools only.</p></body></html>"
    ).encode()
    http = _StubHttp(
        [
            (
                "fullTextXML",
                _response(b"", final_url="https://europepmc.example/xml", status=404),
            ),
            (
                f"/{pmcid}/pdf/",
                _response(b"", final_url=f"{article_url}pdf/", status=404),
            ),
            (article_url, _response(shell, final_url=article_url)),
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("pmcid", pmcid),
        metadata=ScholarMetadata(
            pmcid=pmcid,
            title="Population mental health outcomes",
            authors=["Lewis"],
        ),
        original_url=article_url,
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    assert any(
        attempt.provider == "pmc-direct-html"
        and "too short" in attempt.detail
        for attempt in result.attempts
    )


def test_linked_preprint_rate_limit_is_not_reported_as_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scholar_http_module.time, "sleep", lambda _: None)
    published_doi = "10.1126/science.ade2574"
    preprint_doi = "10.1101/2022.07.20.500902"
    http = _StubHttp(
        [
            (
                "/pubs/biorxiv/",
                _response(
                    json.dumps(
                        {"collection": [{"preprint_doi": preprint_doi}]}
                    ).encode(),
                    final_url=f"https://api.biorxiv.org/pubs/biorxiv/{published_doi}",
                    content_type="application/json",
                ),
            ),
            (
                "/pubs/medrxiv/",
                _response(
                    json.dumps({"collection": []}).encode(),
                    final_url=f"https://api.biorxiv.org/pubs/medrxiv/{published_doi}",
                    content_type="application/json",
                ),
            ),
            (
                f"/details/biorxiv/{preprint_doi}",
                _response(
                    b"",
                    final_url=f"https://api.biorxiv.org/details/biorxiv/{preprint_doi}",
                    status=429,
                    content_type="application/json",
                ),
            ),
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", published_doi),
        metadata=ScholarMetadata(doi=published_doi),
        original_url=f"https://doi.org/{published_doi}",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )
    details = " ".join(attempt.detail for attempt in result.attempts)

    assert sum(f"/details/biorxiv/{preprint_doi}" in call for call in http.calls) == 4
    assert "HTTP 429" in details
    assert "no preprint linked to this DOI" not in details


def test_failed_oa_candidate_falls_through_to_the_next_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    blocked = "https://publisher.example/blocked.pdf"
    working = "https://repository.example/paper.pdf"
    http = _StubHttp(
        [
            (blocked, RetrievalError("HTTP 403")),
            (
                working,
                _response(
                    b"%PDF-1.7\nfixture",
                    final_url=working,
                    content_type="application/pdf",
                ),
            ),
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/example"),
        metadata=ScholarMetadata(
            doi="10.1234/example",
            oa_locations=[
                OALocation(blocked, "pdf", "openalex"),
                OALocation(working, "pdf", "openalex"),
            ],
        ),
        original_url="https://publisher.example/article",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is not None
    assert result.paper.final_url == working
    candidate_calls = [call for call in http.calls if "/pubs/" not in call]
    assert candidate_calls[:2] == [blocked, working]
    assert any("HTTP 403" in attempt.detail for attempt in result.attempts)


def test_unextractable_candidate_also_falls_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    first = "https://repository.example/first.pdf"
    second = "https://repository.example/second.pdf"
    http = _StubHttp(
        [
            (
                first,
                _response(b"%PDF-first", final_url=first, content_type="application/pdf"),
            ),
            (
                second,
                _response(b"%PDF-second", final_url=second, content_type="application/pdf"),
            ),
        ]
    )
    validated: list[str] = []

    def validate(paper: AcquiredPaper) -> tuple[bool, str]:
        validated.append(paper.final_url)
        return (
            (False, "Docling could not extract the first candidate")
            if paper.final_url == first
            else (True, "Extracted")
        )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/extraction-fallthrough"),
        metadata=ScholarMetadata(
            doi="10.1234/extraction-fallthrough",
            oa_locations=[
                OALocation(first, "pdf", "openalex"),
                OALocation(second, "pdf", "openalex"),
            ],
        ),
        original_url="https://publisher.example/article",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
        validate=validate,
    )

    assert result.paper is not None
    assert result.paper.final_url == second
    assert validated == [first, second]
    assert any("Docling" in attempt.detail for attempt in result.attempts)


def test_public_host_guard_rejects_loopback_and_private_candidates() -> None:
    assert not is_public_candidate("http://localhost:4000/bitstreams/paper.pdf")
    assert not is_public_candidate("http://127.0.0.1/paper.pdf")
    assert not is_public_candidate("http://0177.0.0.1/paper.pdf")
    assert not is_public_candidate("http://0x7f000001/paper.pdf")
    assert not is_public_candidate("http://2130706433/paper.pdf")
    assert not is_public_candidate("http://127.1/paper.pdf")
    assert not is_public_candidate("http://10.0.0.8/paper.pdf")
    assert not is_public_candidate("http://169.254.2.1/paper.pdf")
    assert is_public_candidate(
        "https://repository.example/paper.pdf",
        resolve=PUBLIC_RESOLVER,
    )


def test_public_host_guard_blocks_candidate_before_http_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    loopback = "http://localhost:4000/bitstreams/paper.pdf"
    http = _StubHttp(
        [
            (
                loopback,
                _response(
                    b"%PDF-1.7\nmust-not-fetch",
                    final_url=loopback,
                    content_type="application/pdf",
                ),
            )
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/loopback"),
        metadata=ScholarMetadata(
            doi="10.1234/loopback",
            oa_locations=[OALocation(loopback, "pdf", "openalex")],
        ),
        original_url="https://publisher.example/article",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    assert loopback not in http.calls
    assert any("not public" in attempt.detail for attempt in result.attempts)


def test_candidate_redirect_to_private_host_is_blocked_before_second_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    public = "https://repository.example/paper.pdf"
    private = "http://127.0.0.1/internal.pdf"
    http = _StubHttp(
        [
            (
                public,
                _response(
                    b"",
                    final_url=public,
                    status=302,
                    headers={"location": private},
                ),
            ),
            (
                private,
                _response(
                    b"%PDF-private",
                    final_url=private,
                    content_type="application/pdf",
                ),
            ),
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/redirect"),
        metadata=ScholarMetadata(
            doi="10.1234/redirect",
            oa_locations=[OALocation(public, "pdf", "openalex")],
        ),
        original_url="https://publisher.example/article",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    candidate_calls = [call for call in http.calls if "/pubs/" not in call]
    assert candidate_calls == [public]
    public_call = http.calls.index(public)
    assert http.call_options[public_call]["allow_redirects"] is False
    assert isinstance(http.call_options[public_call]["max_bytes"], int)
    assert any(
        attempt.url == private and "not public" in attempt.detail
        for attempt in result.attempts
    )


def test_candidate_redirect_chain_is_bounded_to_three_hops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    urls = [f"https://repository.example/hop-{index}" for index in range(5)]
    http = _StubHttp(
        [
            (
                current,
                _response(
                    b"",
                    final_url=current,
                    status=302,
                    headers={"location": following},
                ),
            )
            for current, following in zip(urls, urls[1:])
        ]
    )

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/redirect-limit"),
        metadata=ScholarMetadata(
            doi="10.1234/redirect-limit",
            oa_locations=[OALocation(urls[0], "pdf", "openalex")],
        ),
        original_url="https://publisher.example/article",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, http),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    candidate_calls = [call for call in http.calls if "/pubs/" not in call]
    assert candidate_calls == urls[:4]
    assert any("3-redirect limit" in attempt.detail for attempt in result.attempts)


def test_recipe_http_client_aborts_response_past_size_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class OversizedResponse:
        status_code = 200
        url = "https://repository.example/huge.pdf"
        headers = {"content-type": "application/pdf", "content-length": "6"}
        content = b""

        def iter_content(self, chunk_size: int):
            yield b"123456"

        def close(self) -> None:
            nonlocal closed
            closed = True

    calls: dict[str, object] = {}

    def fake_get(url: str, **kwargs):
        calls.update({"url": url, **kwargs})
        return OversizedResponse()

    monkeypatch.setattr("omniread.recipes.base.requests.get", fake_get)

    with pytest.raises(RetrievalError, match="exceeds the 5-byte limit"):
        RecipeHttpClient().get(
            "https://repository.example/huge.pdf",
            allow_redirects=False,
            max_bytes=5,
        )

    assert calls["allow_redirects"] is False
    assert calls["stream"] is True
    assert closed is True


def test_repository_landing_exposes_one_hop_pdf_candidate() -> None:
    html = (FIXTURES / "scholar_repo_landing.html").read_text()

    assert citation_pdf_urls(
        html, "https://hal.science/hal-04206682"
    ) == ["https://hal.science/hal-04206682/document"]


def test_malformed_fetched_urls_are_rejected_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    malformed_pdf = "https://a[b.example.org/paper.pdf"
    malformed_landing = "https://repo[sitory.example.org/item/1"
    html = (
        '<meta name="citation_pdf_url" content="http://[">'
        "<body><p>Landing page</p></body>"
    )

    assert citation_pdf_urls(html, "https://publisher.example/article") == []
    assert not is_public_candidate(malformed_pdf, resolve=PUBLIC_RESOLVER)

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1234/malformed-locators"),
        metadata=ScholarMetadata(
            doi="10.1234/malformed-locators",
            oa_locations=[
                OALocation(malformed_pdf, "pdf", "openalex"),
                OALocation(malformed_landing, "landing", "openalex"),
            ],
        ),
        original_url="https://publisher.example/article",
        landing=HttpResponse(
            html.encode(),
            200,
            "https://publisher.example/article",
            {"content-type": "text/html"},
        ),
        opts=ReadOptions(),
        ctx=_context(tmp_path, _StubHttp([])),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    assert any("not public" in attempt.detail for attempt in result.attempts)


def test_malformed_citation_pdf_url_never_raises_from_public_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(locate_module, "resolve_host", PUBLIC_RESOLVER)
    url = "https://www.nature.com/articles/nature14539"
    landing = HttpResponse(
        (
            '<head><meta name="citation_doi" content="10.1038/nature14539">'
            '<meta name="citation_pdf_url" content="http://["></head>'
            "<body><h1>Deep learning</h1><h2>Abstract</h2>"
            "<p>A short abstract.</p></body>"
        ).encode(),
        200,
        url,
        {"content-type": "text/html"},
    )
    http = _StubHttp(
        [
            ("api.openalex.org", RetrievalError("OpenAlex unavailable")),
            ("api.crossref.org", RetrievalError("Crossref unavailable")),
            ("europepmc", RetrievalError("Europe PMC unavailable")),
            (url, landing),
        ]
    )

    result = AcademicRecipe().read(url, ReadOptions(), _context(tmp_path, http))

    assert result.completeness.status == "incomplete"
    assert result.structured_data["scholarly"]["content_level"] in {
        "metadata_only",
        "abstract_only",
    }


@pytest.mark.parametrize(
    "landing_outcome",
    [
        _response(
            "normal_article.html",
            final_url="https://www.nature.com/articles/no-canonical-id",
        ),
        RetrievalError("HTTP 403"),
    ],
    ids=["fetched-without-id", "fetch-failed"],
)
def test_unresolvable_scholarly_page_degrades_to_metadata_only(
    tmp_path: Path,
    landing_outcome: HttpResponse | Exception,
) -> None:
    url = "https://www.nature.com/articles/no-canonical-id"
    result = AcademicRecipe().read(
        url,
        ReadOptions(),
        _context(tmp_path, _StubHttp([(url, landing_outcome)])),
    )

    scholarly = result.structured_data["scholarly"]
    assert scholarly["content_level"] == "metadata_only"
    assert scholarly["identifier"] is None
    assert result.completeness.status == "incomplete"
    assert result.cost_to_complete is not None
    assert "No canonical scholarly identifier" in scholarly["extraction_error"]


def test_unidentifiable_open_article_keeps_generic_page_content(
    tmp_path: Path,
) -> None:
    url = "https://elifesciences.org/articles/00001"
    landing = HttpResponse(
        (
            "<html><head><title>Readable open article</title></head><body>"
            "<h1>Readable open article</h1><h2>Introduction</h2>"
            + "<p>Open body evidence. " * 2_000
            + "</p><h2>Methods</h2>"
            + "<p>Reproducible method detail. " * 2_000
            + "</p><h2>Results</h2>"
            + "<p>Observed result detail. " * 2_000
            + "</p></body></html>"
        ).encode(),
        200,
        url,
        {"content-type": "text/html"},
    )
    generic_markdown = (
        "# Readable open article\n\n## Introduction\n\n"
        + "Open body evidence. " * 2_000
        + "\n\n## Methods\n\n"
        + "Reproducible method detail. " * 2_000
        + "\n\n## Results\n\n"
        + "Observed result detail. " * 2_000
    )

    def generic_read(target: str, **kwargs) -> ReadResult:
        return ReadResult(
            url=target,
            content=generic_markdown,
            outline=build_outline(generic_markdown),
            completeness=Completeness(
                "complete",
                [Evidence("page_capture", True, "The open page was captured")],
                "The page-level representation is complete",
            ),
            provenance=Provenance(
                1,
                "curl_cffi+trafilatura",
                "generic",
                target,
                NOW.isoformat(),
                200,
                target,
                [target],
            ),
            structured_data={},
            coverage="full",
            cost_to_complete=None,
            truncated=False,
            omitted=[],
        )

    result = AcademicRecipe().read(
        url,
        ReadOptions(),
        _context(tmp_path, _StubHttp([(url, landing)]), generic_read=generic_read),
    )

    scholarly = result.structured_data["scholarly"]
    assert scholarly["content_level"] == "metadata_only"
    assert result.completeness.status == "incomplete"
    assert len(result.content) > 50_000
    assert "Observed result detail" in result.content


def test_springer_paywall_is_rejected_cross_publisher_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(locate_module, "resolve_host", PUBLIC_RESOLVER)
    url = "https://link.springer.com/article/10.1007/s11263-015-0816-y"
    response = _response("scholar_springer_paywall.html", final_url=url)

    http = _StubHttp(
        [
            (
                "content/pdf",
                _response("scholar_springer_paywall.html", final_url=url),
            ),
            (url, response),
        ]
    )
    result = AcademicRecipe().read(
        url,
        ReadOptions(),
        _context(tmp_path, http),
    )

    scholarly = result.structured_data["scholarly"]
    assert scholarly["content_level"] == "abstract_only"
    assert result.completeness.status == "incomplete"
    assert "ImageNet Large Scale Visual Recognition Challenge" in result.content
    assert any(
        attempt["succeeded"] is False and "magic bytes" in attempt["detail"]
        for attempt in scholarly["acquisition_attempts"]
    )


def test_stored_profile_springer_fixture_never_becomes_full_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(locate_module, "resolve_host", PUBLIC_RESOLVER)
    url = "https://link.springer.com/article/10.1007/s11263-015-0816-y"
    response = _response("scholar_springer_paywall.html", final_url=url)
    extracted = extract_page(response.text, url=url)
    http = _StubHttp(
        [
            (
                "api.openalex.org",
                _response(
                    json.dumps(
                        {
                            "doi": "https://doi.org/10.1007/s11263-015-0816-y",
                            "title": "ImageNet Large Scale Visual Recognition Challenge",
                            "biblio": {"first_page": "199", "last_page": "212"},
                        }
                    ).encode(),
                    final_url="https://api.openalex.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            (
                "api.crossref.org",
                _response(
                    json.dumps(
                        {
                            "message": {
                                "DOI": "10.1007/s11263-015-0816-y",
                                "title": [
                                    "ImageNet Large Scale Visual Recognition Challenge"
                                ],
                                "page": "199-212",
                            }
                        }
                    ).encode(),
                    final_url="https://api.crossref.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            ("europepmc", RetrievalError("No matching Europe PMC record")),
            (
                "content/pdf",
                _response("scholar_springer_paywall.html", final_url=url),
            ),
            (url, response),
        ]
    )
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir(url).mkdir(parents=True)

    def generic_read(target: str, **kwargs) -> ReadResult:
        return ReadResult(
            url=target,
            content=extracted.markdown,
            outline=build_outline(extracted.markdown),
            completeness=Completeness(
                "complete",
                [
                    Evidence("page_capture", True, "The paywalled page was captured"),
                    Evidence("dom_coverage", True, "The visible page was captured"),
                ],
                "The page-level capture is complete",
            ),
            provenance=Provenance(
                4,
                "persistent-profile",
                "generic",
                target,
                NOW.isoformat(),
                200,
                target,
                [target],
            ),
            structured_data=extracted.structured_data,
            coverage="full",
            cost_to_complete=None,
            truncated=False,
            omitted=[],
        )

    result = AcademicRecipe().read(
        url,
        ReadOptions(),
        RecipeContext(
            clock=lambda: NOW,
            generic_read=generic_read,
            http=http,  # type: ignore[arg-type]
            profiles=profiles,
            render=None,
        ),
    )

    scholarly = result.structured_data["scholarly"]
    assert scholarly["content_level"] == "abstract_only"
    assert scholarly["full_text_acquisition"] is None
    assert result.completeness.status == "incomplete"
    assert any(
        attempt["rung"] == 8
        and attempt["succeeded"] is False
        and "paper body" in attempt["detail"]
        for attempt in scholarly["acquisition_attempts"]
    )


def test_paywalled_publisher_recipe_returns_honest_abstract_not_an_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(locate_module, "resolve_host", PUBLIC_RESOLVER)
    url = "https://www.nature.com/articles/nature14539"
    http = _nature_http()

    result = AcademicRecipe().read(
        url,
        ReadOptions(),
        _context(tmp_path, http),
    )

    scholarly = result.structured_data["scholarly"]
    assert scholarly["content_level"] == "abstract_only"
    assert result.completeness.status == "incomplete"
    assert result.cost_to_complete is not None
    assert result.cost_to_complete.required_tier == 4
    assert scholarly["unlock_path"] == "omniread login nature.com"
    assert "Deep learning allows computational models" in result.content
    assert any(
        "magic bytes" in attempt["detail"]
        for attempt in scholarly["acquisition_attempts"]
    )


def test_stored_publisher_profile_with_zero_of_two_reference_manifests_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(locate_module, "resolve_host", PUBLIC_RESOLVER)
    url = "https://www.nature.com/articles/nature14539"
    http = _nature_http()
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir(url).mkdir(parents=True)
    calls: list[dict[str, object]] = []
    markdown = (
        "# Deep learning\n\n## Introduction\n\n"
        + " ".join(f"body{i}" for i in range(2_000))
        + "\n\n## Conclusion\n\nThe body concludes."
    )

    def generic_read(target: str, **kwargs) -> ReadResult:
        calls.append({"target": target, **kwargs})
        return ReadResult(
            url=target,
            content=markdown,
            outline=build_outline(markdown),
            completeness=Completeness(
                "unknown",
                [Evidence("page_level", None, "Scholar verifier decides the artifact")],
                "Page-level verdict is not the scholarly verdict",
            ),
            provenance=Provenance(
                4,
                "persistent-profile",
                "generic",
                target,
                NOW.isoformat(),
                200,
                target,
                [target],
            ),
            structured_data={},
            coverage="full",
            cost_to_complete=None,
            truncated=False,
            omitted=[],
        )

    ctx = RecipeContext(
        clock=lambda: NOW,
        generic_read=generic_read,
        http=http,  # type: ignore[arg-type]
        profiles=profiles,
        render=None,
    )
    result = AcademicRecipe().read(url, ReadOptions(), ctx)

    # Rung 8 reads the publisher page anonymously first; the generic ladder still
    # escalates to the stored profile on its own, so an anonymous win never has to
    # be re-fetched behind the login.
    assert calls and calls[0].get("auth_required") is not True
    evidence = {item.name: item for item in result.completeness.evidence}
    assert result.structured_data["scholarly"]["content_level"] == "full_text"
    assert result.completeness.status == "incomplete"
    assert evidence["section_manifest"].passed is True
    assert evidence["abstract_containment"].passed is None
    assert evidence["page_extent"].passed is None
    assert evidence["reference_manifest"].passed is False
    assert result.provenance.tier == 4


def test_stored_profile_that_still_returns_real_paywall_stays_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(locate_module, "resolve_host", PUBLIC_RESOLVER)
    url = "https://www.nature.com/articles/nature14539"
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir(url).mkdir(parents=True)
    raw_html = (FIXTURES / "scholar_nature_paywall.html").read_text()
    markdown = extract_html(
        raw_html,
        "https://www.nature.com/articles/nature14539",
    )

    def generic_read(target: str, **kwargs) -> ReadResult:
        return ReadResult(
            url=target,
            content=markdown.markdown,
            outline=build_outline(markdown.markdown),
            completeness=Completeness(
                "complete",
                [
                    Evidence("page_capture", True, "The abstract page was captured"),
                    Evidence("dom_coverage", True, "The visible page was captured"),
                ],
                "The page-level capture is complete",
            ),
            provenance=Provenance(
                4,
                "persistent-profile",
                "generic",
                target,
                NOW.isoformat(),
                200,
                target,
                [target],
            ),
            structured_data={},
            coverage="full",
            cost_to_complete=None,
            truncated=False,
            omitted=[],
        )

    result = AcademicRecipe().read(
        url,
        ReadOptions(),
        RecipeContext(
            clock=lambda: NOW,
            generic_read=generic_read,
            http=_nature_http(),  # type: ignore[arg-type]
            profiles=profiles,
            render=None,
        ),
    )

    assert result.structured_data["scholarly"]["content_level"] == "abstract_only"
    assert result.structured_data["scholarly"]["full_text_acquisition"] is None
    assert result.completeness.status == "incomplete"
    assert result.cost_to_complete and result.cost_to_complete.required_tier == 4
    attempts = result.structured_data["scholarly"]["acquisition_attempts"]
    assert any(
        attempt["rung"] == 8
        and attempt["succeeded"] is False
        and "paper body" in attempt["detail"]
        for attempt in attempts
    )


def test_abstract_only_can_never_pair_with_complete_for_any_input_shape() -> None:
    abstract = "A concise abstract with enough bibliographic meaning."
    abstract_extraction = ScholarExtraction(
        "# Paper\n\n## Abstract\n\n" + abstract,
        ("Abstract",),
        0,
        False,
        len(abstract.split()),
    )
    acquired = AcquiredPaper(
        "html",
        b"",
        8,
        "authenticated",
        "https://publisher.example/paper",
        200,
        ("https://publisher.example/paper",),
    )

    for extraction, acquisition, known_abstract in product(
        (None, abstract_extraction),
        (None, acquired),
        (None, abstract),
    ):
        verdict = verify_scholarly(
            extraction=extraction,
            metadata=ScholarMetadata(abstract=known_abstract),
            acquisition=acquisition,
        )
        if verdict.content_level == "abstract_only":
            assert verdict.completeness.status == "incomplete"
        assert not (
            verdict.content_level == "abstract_only"
            and verdict.completeness.status == "complete"
        )


def test_preprint_with_only_one_reference_line_cannot_claim_complete() -> None:
    markdown = (
        "# Preprint\n\n## Introduction\n\n"
        + "body " * 1_500
        + "\n\n## Methods\n\n"
        + "method " * 1_500
        + "\n\n## References\n\n1. One detected reference"
    )
    extraction = ScholarExtraction(
        markdown,
        ("Introduction", "Methods", "References"),
        1,
        False,
        len(markdown.split()),
    )
    paper = AcquiredPaper(
        "html",
        markdown.encode(),
        3,
        "arxiv-html",
        "https://arxiv.org/html/2401.00001",
        200,
        ("https://arxiv.org/html/2401.00001",),
    )

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(arxiv_id="2401.00001"),
        acquisition=paper,
    )
    evidence = {item.name: item for item in verdict.completeness.evidence}

    assert verdict.content_level == "full_text"
    assert evidence["section_manifest"].passed is True
    assert evidence["page_extent"].passed is None
    assert evidence["reference_manifest"].passed is None
    assert verdict.completeness.status == "unknown"


def test_zero_references_against_two_positive_manifests_vetoes_false_complete() -> None:
    extraction = ScholarExtraction(
        "# Paper\n\n## Methods\n\n"
        + "evidence " * 1_000
        + "\n\n## Results\n\n"
        + "finding " * 1_000,
        ("Methods", "Results"),
        0,
        False,
        2_005,
    )
    paper = AcquiredPaper(
        "pdf",
        b"%PDF",
        4,
        "openalex",
        "https://repository.example/paper.pdf",
        200,
        ("https://repository.example/paper.pdf",),
    )
    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(
            first_page="1",
            last_page="5",
            crossref_reference_count=100,
            openalex_reference_count=80,
        ),
        acquisition=paper,
    )
    evidence = {item.name: item for item in verdict.completeness.evidence}

    assert evidence["reference_manifest"].passed is False
    assert evidence["section_manifest"].passed is True
    assert evidence["page_extent"].passed is True
    assert verdict.completeness.status == "incomplete"


def _nature_http() -> _StubHttp:
    url = "https://www.nature.com/articles/nature14539"
    return _StubHttp(
        [
            (
                "nature14539.pdf",
                _response(
                    "scholar_pdf_masquerade.html",
                    final_url=f"{url}.pdf",
                ),
            ),
            (
                "api.openalex.org",
                _response(
                    "scholar_openalex_paywalled.json",
                    final_url="https://api.openalex.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            (
                "api.crossref.org",
                _response(
                    "scholar_crossref_paywalled.json",
                    final_url="https://api.crossref.org/works/fixture",
                    content_type="application/json",
                ),
            ),
            ("europepmc/webservices/rest/search", RetrievalError("No matching EPMC record")),
            (
                "hal-04206682/document",
                RetrievalError("Repository PDF returned HTTP 403"),
            ),
            (
                "hal.science/hal-04206682",
                _response(
                    "scholar_repo_landing.html",
                    final_url="https://hal.science/hal-04206682",
                ),
            ),
            (
                url,
                _response("scholar_nature_paywall.html", final_url=url),
            ),
        ]
    )


def test_unlock_path_never_points_at_a_doi_redirector() -> None:
    """A DOI that never resolved must not send the caller to log in to doi.org.

    Live-observed on 2026-07-28: reading 10.1111/j.1467-8624.2010.01564.x and
    10.2307/1914185 both emitted `omniread login doi.org`. doi.org forwards to a
    publisher, it has no account and no paywall, so that instruction cannot unlock
    anything. The target must be a publisher or repository host, or absent.
    """

    redirector_only = ScholarMetadata(doi="10.2307/1914185")
    assert (
        locate_module._publisher_target(
            landing=None,
            metadata=redirector_only,
            original_url="https://doi.org/10.2307/1914185",
        )
        is None
    )

    with_publisher_landing = ScholarMetadata(doi="10.1111/j.1467-8624.2010.01564.x")
    with_publisher_landing.oa_locations.append(
        OALocation("https://onlinelibrary.wiley.com/doi/10.1111/x", "landing", "openalex")
    )
    assert (
        locate_module._publisher_target(
            landing=None,
            metadata=with_publisher_landing,
            original_url="https://doi.org/10.1111/j.1467-8624.2010.01564.x",
        )
        == "https://onlinelibrary.wiley.com/doi/10.1111/x"
    )

    # A publisher host that merely starts with the letters of a redirector is fine.
    assert not locate_module._is_redirector("https://wiley.com/article/1")
    assert not locate_module._is_redirector("https://doi.example.com/x")
    assert locate_module._is_redirector("https://dx.doi.org/10.1/x")
    assert locate_module._is_redirector("https://WWW.DOI.ORG/10.1/x")


# --- Free to read at the publisher, open access nowhere ---------------------
#
# The 2026-08-18 miss: BJSM 2025;59(2):78-90 (PMID 39638438) is marked FREE on
# bjsm.bmj.com and carries its whole body there, while Unpaywall, OpenAlex,
# Crossref and Europe PMC all report "subscription required". Every rung keyed on
# an OA index therefore found nothing and the paper was reported paywalled.


def _publisher_read_result(target: str, *, body_words: int = 2_000, tier: int = 1) -> ReadResult:
    markdown = (
        "# International Delphi consensus on bone stress injuries in athletes\n\n"
        "Tim Hoenig, Karsten Hollander, Adam S Tenforde\n\n## Methods\n\n"
        + " ".join(f"body{index}" for index in range(body_words))
        + "\n\n## Results\n\nConsensus was reached on 41 of 58 statements."
    )
    return ReadResult(
        url=target,
        content=markdown,
        outline=build_outline(markdown),
        completeness=Completeness(
            "unknown",
            [Evidence("page_level", None, "The scholar verifier decides the artifact")],
            "Page-level verdict is not the scholarly verdict",
        ),
        provenance=Provenance(
            tier,
            "curl_cffi+trafilatura",
            "generic",
            target,
            NOW.isoformat(),
            200,
            "https://bjsm.bmj.com/content/59/2/78",
            [target, "https://bjsm.bmj.com/content/59/2/78"],
        ),
        structured_data={},
        coverage="full",
        cost_to_complete=None,
        truncated=False,
        omitted=[],
    )


def _delphi_metadata() -> ScholarMetadata:
    return ScholarMetadata(
        doi="10.1136/bjsports-2024-108616",
        title="International Delphi consensus on bone stress injuries in athletes",
        authors=["Hoenig", "Hollander", "Tenforde"],
    )


def test_free_to_read_publisher_page_is_acquired_when_no_index_reports_open_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    calls: list[dict[str, object]] = []

    def generic_read(target: str, **kwargs) -> ReadResult:
        calls.append({"target": target, **kwargs})
        return _publisher_read_result(target)

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("pmid", "39638438"),
        metadata=_delphi_metadata(),
        original_url="https://pubmed.ncbi.nlm.nih.gov/39638438/",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, _StubHttp([]), generic_read=generic_read),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is not None
    assert result.paper.provider == "anonymous-publisher-render"
    # The caller only ever supplied a PubMed URL; the DOI is what reaches the publisher.
    assert calls[0]["target"] == "https://doi.org/10.1136/bjsports-2024-108616"
    assert calls[0].get("auth_required") is not True
    # An anonymous read is not a login, and must not be provenanced as one.
    assert result.paper.tier == 1


def test_publisher_page_read_anonymously_must_still_prove_its_own_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)

    def generic_read(target: str, **kwargs) -> ReadResult:
        return _publisher_read_result(target)

    rejected: list[str] = []

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1136/bjsports-2024-108616"),
        metadata=_delphi_metadata(),
        original_url="https://doi.org/10.1136/bjsports-2024-108616",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, _StubHttp([]), generic_read=generic_read),
        resolve=PUBLIC_RESOLVER,
        validate=lambda paper: (rejected.append(paper.provider), (False, "rejected"))[1],
    )

    assert result.paper is None
    assert rejected == ["anonymous-publisher-render"]
    assert not result.paper


def test_unlock_path_names_the_publisher_the_render_landed_on_not_the_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)

    def generic_read(target: str, **kwargs) -> ReadResult:
        # A real paywall: the publisher page resolves but exposes no body.
        empty = _publisher_read_result(target, body_words=0)
        return replace_result_content(empty, "")

    result = acquire_full_text(
        identifier=ScholarlyIdentifier("pmid", "39638438"),
        metadata=_delphi_metadata(),
        original_url="https://pubmed.ncbi.nlm.nih.gov/39638438/",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, _StubHttp([]), generic_read=generic_read),
        resolve=PUBLIC_RESOLVER,
    )

    assert result.paper is None
    # Before the fix this read "omniread login nih.gov": an index has no paywall,
    # so the reader was sent somewhere that could never unlock anything.
    assert result.untried_unlocks == ["omniread login bmj.com"]


def replace_result_content(result: ReadResult, content: str) -> ReadResult:
    return ReadResult(
        url=result.url,
        content=content,
        outline=build_outline(content),
        completeness=result.completeness,
        provenance=result.provenance,
        structured_data={},
        coverage="full",
        cost_to_complete=None,
        truncated=False,
        omitted=[],
    )


def test_bmj_journal_subdomains_route_to_the_scholarly_recipe() -> None:
    # Every BMJ journal lives on its own subdomain; only the apex was routed, so
    # a BJSM article silently fell through to the generic recipe and never ran the
    # OA ladder at all.
    assert is_scholarly_input("https://bjsm.bmj.com/content/59/2/78")
    assert is_scholarly_input("https://gut.bmj.com/content/74/1/1")
    assert is_scholarly_input("https://www.bmj.com/content/380/bmj-2022-072385")
    # The article-path guard still keeps non-article BMJ subdomains out.
    assert not is_scholarly_input("https://blogs.bmj.com/bjsm/2025/01/01/some-post/")
    assert not is_scholarly_input("https://bjsm.bmj.com/pages/authors/")


def test_repository_landing_never_preempts_the_doi_as_the_publisher_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIREAD_CONTACT_EMAIL", raising=False)
    metadata = _delphi_metadata()
    # OpenAlex commonly lists an institutional repository record first. Rung 7 has
    # already mined those; reading one as "the publisher" reported the paper as
    # unlockable by logging in to the repository.
    metadata.oa_locations.append(
        OALocation("https://hal.science/hal-04206682", "landing", "openalex:repository")
    )
    targets: list[str] = []

    def generic_read(target: str, **kwargs) -> ReadResult:
        targets.append(target)
        return replace_result_content(_publisher_read_result(target), "")

    acquire_full_text(
        identifier=ScholarlyIdentifier("doi", "10.1136/bjsports-2024-108616"),
        metadata=metadata,
        original_url="https://doi.org/10.1136/bjsports-2024-108616",
        landing=None,
        opts=ReadOptions(),
        ctx=_context(tmp_path, _StubHttp([]), generic_read=generic_read),
        resolve=PUBLIC_RESOLVER,
    )

    assert targets == ["https://doi.org/10.1136/bjsports-2024-108616"]


def test_missing_pdf_extra_surfaces_install_path_from_the_recipe(monkeypatch, tmp_path):
    import sys
    from omniread.scholar import recipe as recipe_module
    monkeypatch.setitem(sys.modules, 'pypdf', None)
    monkeypatch.setattr(locate_module, 'resolve_host', PUBLIC_RESOLVER)
    monkeypatch.delenv('OMNIREAD_CONTACT_EMAIL', raising=False)
    pdf_url = 'https://repository.example.test/paper.pdf'
    metadata = ScholarMetadata(
        doi='10.1000/fixture', title='A synthetic paper', authors=['Fixture Author'],
        abstract='A synthetic metadata abstract.',
        oa_locations=[OALocation(pdf_url, 'pdf', 'openalex')],
    )
    monkeypatch.setattr(recipe_module, 'collect_metadata', lambda *_: metadata)
    http = _StubHttp([(pdf_url, _response(_minimal_text_pdf(['Fixture Author', 'A synthetic paper']), final_url=pdf_url, content_type='application/pdf'))])
    result = AcademicRecipe().read('https://doi.org/10.1000/fixture', ReadOptions(), _context(tmp_path, http))
    assert result.completeness.status != 'complete'
    assert '[pdf]' in result.structured_data['scholarly']['extraction_error']
    assert 'pdf extra' in result.structured_data['scholarly']['unlock_path']
