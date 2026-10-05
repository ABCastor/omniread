from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from omniread._html import has_pagination
from omniread.auth import ProfileStore
from omniread.extract import ExtractedPage, extract_page
from omniread.fetch import FetchedPage, detect_block_page
from omniread.recipes.academic import AcademicRecipe
from omniread.recipes.base import HttpResponse, ReadOptions, RecipeContext
from omniread.recipes.facebook_marketplace import (
    build_markdown as _fb_markdown,
    extract_listing as _fb_extract_listing,
    verify_listing as _fb_verify_listing,
)
from omniread.recipes.reddit import (
    _thread_markdown,
    parse_redlib_thread,
    verify_comment_capture,
)
from omniread.scholar.extract import (
    ScholarExtraction,
    extract_abstract_from_html,
    extract_html as extract_scholar_html,
    extract_jats,
)
from omniread.scholar import locate as locate_module
from omniread.scholar.locate import AcquiredPaper
from omniread.scholar.metadata import ScholarMetadata
from omniread.scholar.verify import verify_scholarly
from omniread.types import RetrievalError
from omniread.verify import verify_completeness

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


class _CorpusScholarHttp:
    def get(self, url: str, **kwargs) -> HttpResponse:
        if "api.openalex.org" in url:
            return _fixture_response(
                "scholar_openalex_paywalled.json",
                "https://api.openalex.org/works/fixture",
                "application/json",
            )
        if "api.crossref.org" in url:
            return _fixture_response(
                "scholar_crossref_paywalled.json",
                "https://api.crossref.org/works/fixture",
                "application/json",
            )
        if "europepmc" in url:
            raise RetrievalError("No matching EPMC record")
        if "nature14539.pdf" in url:
            return _fixture_response(
                "scholar_pdf_masquerade.html",
                "https://www.nature.com/articles/nature14539.pdf",
                "text/html",
            )
        if "hal-04206682/document" in url:
            raise RetrievalError("Repository PDF returned HTTP 403")
        if "hal.science/hal-04206682" in url:
            return _fixture_response(
                "scholar_repo_landing.html",
                "https://hal.science/hal-04206682",
                "text/html",
            )
        if "nature.com/articles/nature14539" in url:
            return _fixture_response(
                "scholar_nature_paywall.html",
                "https://www.nature.com/articles/nature14539",
                "text/html",
            )
        raise RetrievalError(f"No corpus route for {url}")


def _fixture_response(name: str, final_url: str, content_type: str) -> HttpResponse:
    return HttpResponse(
        body=(FIXTURES / name).read_bytes(),
        status=200,
        final_url=final_url,
        headers={"content-type": content_type},
    )


def _fetched(raw_html: str, *, url: str = "https://example.test/article") -> FetchedPage:
    return FetchedPage(
        raw_html=raw_html,
        http_status=200,
        final_url=url,
        headers={"content-type": "text/html"},
        block_signal=detect_block_page(raw_html, http_status=200),
    )


def _evaluate(file_name: str):
    raw_html = (FIXTURES / file_name).read_text(encoding="utf-8")
    fetched = _fetched(raw_html)
    extracted = extract_page(raw_html, url=fetched.final_url)
    return extracted, verify_completeness(fetched, extracted)


def _corpus() -> list[dict[str, object]]:
    manifest = json.loads((FIXTURES / "corpus.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    return manifest["cases"]


def _evaluate_case(case: dict[str, object], tmp_path: Path | None = None):
    if case.get("recipe") == "scholar":
        file_name = str(case["file"])
        payload = (FIXTURES / file_name).read_bytes()
        representation = str(case["representation"])
        if representation == "jats":
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
            metadata = ScholarMetadata()
        elif representation == "html":
            url = "https://arxiv.org/html/1706.03762"
            extraction = extract_scholar_html(payload.decode(), url)
            paper = AcquiredPaper(
                "html", payload, 3, "arxiv-html", url, 200, (url,)
            )
            metadata = ScholarMetadata(arxiv_id="1706.03762")
        else:
            assert tmp_path is not None
            url = "https://www.nature.com/articles/nature14539"
            result = AcademicRecipe().read(
                url,
                ReadOptions(),
                RecipeContext(
                    clock=lambda: NOW,
                    # Rung 8 reads the publisher page anonymously before any login;
                    # an authenticated call without a stored profile is still a bug.
                    generic_read=lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError("No stored publisher profile exists")
                        if kwargs.get("auth_required")
                        else RetrievalError("the publisher page is unreachable here")
                    ),
                    http=_CorpusScholarHttp(),  # type: ignore[arg-type]
                    profiles=ProfileStore(tmp_path / "corpus-profiles"),
                    render=None,
                ),
            )
            scholarly = result.structured_data["scholarly"]
            return (
                ExtractedPage(
                    markdown=result.content,
                    structured_data={"content_level": scholarly["content_level"]},
                    canonical_url=result.provenance.canonical_url,
                ),
                result.completeness,
            )
        verdict = verify_scholarly(
            extraction=extraction,
            metadata=metadata,
            acquisition=paper,
        )
        markdown = (
            extraction.markdown
            if extraction is not None
            else f"# {metadata.title}\n\n## Abstract\n\n{metadata.abstract}"
        )
        return (
            ExtractedPage(
                markdown=markdown,
                structured_data={"content_level": verdict.content_level},
                canonical_url=paper.final_url if paper else "https://doi.org/10.1038/nature14539",
            ),
            verdict.completeness,
        )
    if case.get("recipe") == "facebook_marketplace":
        raw_html = (FIXTURES / str(case["file"])).read_text(encoding="utf-8")
        fields = _fb_extract_listing(raw_html, str(case["item_id"]))
        return (
            ExtractedPage(
                markdown=_fb_markdown(fields),
                structured_data=None,
                canonical_url="https://www.facebook.com/marketplace/item/fixture",
            ),
            _fb_verify_listing(fields),
        )
    if case.get("recipe") == "reddit":
        raw_html = (FIXTURES / str(case["file"])).read_text(encoding="utf-8")
        thread = parse_redlib_thread(
            raw_html,
            reddit_url="https://www.reddit.com/r/omniread/comments/abc123/fixture/",
        )
        markdown = _thread_markdown(thread, list(thread.comments)) if thread.comments else ""
        return (
            ExtractedPage(
                markdown=markdown,
                structured_data=None,
                canonical_url="https://www.reddit.com/r/omniread/comments/abc123/fixture/",
            ),
            verify_comment_capture(thread),
        )
    return _evaluate(str(case["file"]))


@pytest.mark.parametrize("case", _corpus(), ids=lambda case: str(case["name"]))
def test_gold_corpus_verdicts(
    case: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(locate_module, "resolve_host", lambda host: ("93.184.216.34",))
    extracted, verdict = _evaluate_case(case, tmp_path)

    assert verdict.status == case["expected_status"]
    if "expected_content_level" in case:
        assert extracted.structured_data["content_level"] == case["expected_content_level"]
    expected_present = case.get("present_sections", case["required_sections"])
    for heading in expected_present:  # type: ignore[union-attr]
        assert str(heading) in extracted.markdown


def test_deliberately_truncated_fixture_is_incomplete() -> None:
    _, verdict = _evaluate("truncated_article.html")
    evidence = {item.name: item for item in verdict.evidence}

    assert verdict.status == "incomplete"
    assert evidence["length_sanity"].passed is False
    assert evidence["blockpage_check"].passed is True


def test_http_200_block_fixture_is_unknown_never_complete() -> None:
    _, verdict = _evaluate("block_page.html")
    evidence = {item.name: item for item in verdict.evidence}

    assert verdict.status == "unknown"
    assert evidence["blockpage_check"].passed is False


def test_page_one_with_explicit_pagination_is_incomplete() -> None:
    _, verdict = _evaluate("paginated_thread.html")
    evidence = {item.name: item for item in verdict.evidence}

    assert verdict.status == "incomplete"
    assert evidence["pagination_coverage"].passed is False
    assert "only one page" in evidence["pagination_coverage"].detail


def test_sequential_docs_next_link_is_not_content_pagination() -> None:
    raw_html = (FIXTURES / "docs_sequential_next.html").read_text(encoding="utf-8")
    url = "https://docs.example.test/library/asyncio-task.html"
    fetched = _fetched(raw_html, url=url)
    extracted = extract_page(raw_html, url=url)

    verdict = verify_completeness(fetched, extracted)
    evidence = {item.name: item for item in verdict.evidence}

    assert has_pagination(raw_html, current_url=url) is False
    assert verdict.status == "complete"
    assert evidence["pagination_coverage"].passed is None


@pytest.mark.parametrize(
    ("current_url", "href"),
    [
        ("https://example.test/thread", "?page=2"),
        ("https://example.test/thread", "/thread/page/2"),
        ("https://example.test/thread.html", "/thread-2.html"),
    ],
)
def test_same_resource_rel_next_is_content_pagination(current_url: str, href: str) -> None:
    raw_html = f'<html><head><link rel="next" href="{href}"></head><body></body></html>'

    assert has_pagination(raw_html, current_url=current_url) is True


def test_false_complete_rate_over_starter_corpus_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(locate_module, "resolve_host", lambda host: ("93.184.216.34",))
    outcomes = []
    for case in _corpus():
        _, verdict = _evaluate_case(case, tmp_path)
        outcomes.append((str(case["expected_status"]), verdict.status))

    adversarial = [outcome for outcome in outcomes if outcome[0] != "complete"]
    false_completes = sum(actual == "complete" for _, actual in adversarial)
    false_complete_rate = false_completes / len(adversarial)

    assert false_complete_rate == 0.0


def test_large_extraction_gap_is_incomplete_even_with_full_raw_dom() -> None:
    raw_html = (FIXTURES / "normal_article.html").read_text(encoding="utf-8")
    fetched = _fetched(raw_html)
    full = extract_page(raw_html, url=fetched.final_url)
    partial = ExtractedPage(
        markdown="# Measuring a Reliable Reader\n\nA reliable reader must distinguish fetching.",
        structured_data=full.structured_data,
        canonical_url=full.canonical_url,
    )

    verdict = verify_completeness(fetched, partial)

    assert verdict.status == "incomplete"
    failed = {item.name for item in verdict.evidence if item.passed is False}
    assert {"heading_presence", "dom_vs_output_coverage", "length_sanity"} <= failed


def test_small_heading_gap_does_not_veto_strong_coverage_signals() -> None:
    raw_html = (FIXTURES / "normal_article.html").read_text(encoding="utf-8")
    fetched = _fetched(raw_html)
    full = extract_page(raw_html, url=fetched.final_url)
    missing_one_heading = ExtractedPage(
        markdown=full.markdown.replace("## Operational limits\n\n", ""),
        structured_data=full.structured_data,
        canonical_url=full.canonical_url,
    )

    verdict = verify_completeness(fetched, missing_one_heading)
    evidence = {item.name: item for item in verdict.evidence}

    assert verdict.status == "complete"
    assert evidence["heading_presence"].passed is None
    assert "ratio 0.80" in evidence["heading_presence"].detail
    assert "missing examples: Operational limits" in evidence["heading_presence"].detail
    assert evidence["dom_vs_output_coverage"].passed is True
    assert evidence["length_sanity"].passed is True


def test_thin_but_plausible_page_fails_closed_as_unknown() -> None:
    raw_html = """
    <html><head><title>Short note</title></head><body><main>
      <h1>Short note</h1><p>This may be the whole note, but there is no independent manifest.</p>
    </main></body></html>
    """
    fetched = _fetched(raw_html)
    extracted = ExtractedPage(
        markdown="# Short note\n\nThis may be the whole note, but there is no independent manifest.",
        structured_data=None,
        canonical_url=fetched.final_url,
    )

    verdict = verify_completeness(fetched, extracted)

    assert verdict.status == "unknown"
    assert "too thin" in verdict.reason


def test_headings_rendered_as_text_not_markdown_are_not_incomplete() -> None:
    # Real docs pages (e.g. the Python docs) render section titles as non-heading text and
    # carry a "¶" permalink glyph, so the extract has no matching Markdown headings even
    # though every section's content is present. That must NOT read as a content gap.
    sections = [f"Configuration Section {i}" for i in range(1, 8)]
    raw_html = (
        "<html><head><title>Guide</title></head><body><main>"
        + "".join(
            f"<h2>{title} ¶</h2><p>"
            + " ".join(f"para{i}word{j}" for j in range(40))
            + "</p>"
            for i, title in enumerate(sections)
        )
        + "</main></body></html>"
    )
    fetched = _fetched(raw_html)
    # The extract keeps every title as PLAIN text (no leading '#') plus its full paragraph.
    markdown = "\n\n".join(
        f"{title}\n\n" + " ".join(f"para{i}word{j}" for j in range(40))
        for i, title in enumerate(sections)
    )
    extracted = ExtractedPage(
        markdown=markdown, structured_data=None, canonical_url=fetched.final_url
    )

    verdict = verify_completeness(fetched, extracted)
    evidence = {item.name: item for item in verdict.evidence}

    assert verdict.status != "incomplete"
    assert evidence["heading_presence"].passed is True


def _headingless_pdf_extraction(word_count: int) -> ScholarExtraction:
    body = " ".join(f"bodyword{i}" for i in range(word_count))
    return ScholarExtraction(
        markdown=f"# Deep learning\n\n{body}",
        section_titles=(),
        reference_count=0,
        references_resolved=False,
        word_count=word_count,
    )


def _headingless_pdf_metadata() -> ScholarMetadata:
    abstract = " ".join(f"abstractword{i}" for i in range(120))
    return ScholarMetadata(
        doi="10.1038/nature14539",
        title="Deep learning",
        abstract=abstract,
        first_page="436",
        last_page="444",
    )


def _pdf_paper() -> AcquiredPaper:
    return AcquiredPaper(
        "pdf",
        b"%PDF-1.4\n" + b"x" * 64,
        9,
        "provider.example.test",
        "https://provider.example.test/10.1038/nature14539",
        200,
        ("https://provider.example.test/10.1038/nature14539",),
        True,
    )


def test_headingless_pdf_with_corroborating_extent_reads_full_text_unknown() -> None:
    # Accepted-article and scanned PDFs hide every section heading from the text
    # layer. When the page range corroborates the extracted word count (ADR 0008),
    # the body is proven structurally and the verdict must be unknown, never
    # incomplete: there is no evidence of missing content, only of unverifiable
    # section structure.
    extraction = _headingless_pdf_extraction(word_count=2_100)

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=_headingless_pdf_metadata(),
        acquisition=_pdf_paper(),
    )

    assert verdict.content_level == "full_text"
    assert verdict.completeness.status == "unknown"
    section = {item.name: item for item in verdict.completeness.evidence}["section_manifest"]
    assert section.passed is False


def test_headingless_pdf_under_extent_does_not_claim_a_body() -> None:
    # The same hidden-headings artifact truncated to far below the page-range
    # expectation must fail closed: extent does not corroborate, so there is no
    # body proof and no misleading full_text label.
    extraction = _headingless_pdf_extraction(word_count=220)

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=_headingless_pdf_metadata(),
        acquisition=_pdf_paper(),
    )

    assert verdict.content_level != "full_text"
    assert verdict.completeness.status != "complete"


def test_headingless_reference_count_uses_the_highest_seen_index() -> None:
    # PDF text extraction merges several reference entries onto one line, so the
    # faithful count of a heading-less reference list is its highest index.
    from omniread.scholar.extract import _markdown_reference_count

    markdown = (
        "# Paper\n\nbody text [1] with an inline citation\n\n"
        "[1] a) A. Author, J. Journal 2020, 1, 1; b) B. Author, J. J. 2021, 2, 2\n"
        "[2] C. Author, J. J. 2022, 3, 3\n"
        "[99] Z. Author, J. J. 2023, 4, 4"
    )

    assert _markdown_reference_count(markdown) == 99


def test_pdf_with_sections_and_refs_but_short_extent_is_incomplete() -> None:
    # A stub PDF that carries two plausible sections and a reference list but
    # claims far more pages than its words can cover must fail closed: page
    # extent is a PDF amount-veto that no crafted structure can override.
    extraction = ScholarExtraction(
        markdown=(
            "# Paper\n\n## Methods\n\n"
            + "method " * 500
            + "\n\n## Results\n\n"
            + "result " * 500
        ),
        section_titles=("Methods", "Results"),
        reference_count=50,
        references_resolved=False,
        word_count=1_005,
    )
    paper = AcquiredPaper(
        "pdf",
        b"%PDF-1.4\n" + b"x" * 64,
        9,
        "provider.example.test",
        "https://provider.example.test/10.1038/nature14539",
        200,
        ("https://provider.example.test/10.1038/nature14539",),
        True,
    )

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(
            title="Deep learning",
            abstract="A summary of the work.",
            first_page="1",
            last_page="20",
            crossref_reference_count=60,
            openalex_reference_count=60,
        ),
        acquisition=paper,
    )

    assert verdict.content_level == "full_text"
    assert verdict.completeness.status == "incomplete"
    extent = {item.name: item for item in verdict.completeness.evidence}["page_extent"]
    assert extent.passed is False


def test_html_artifact_cannot_claim_a_body_through_extent() -> None:
    # The extent fallback is PDF-only (ADR 0008). A wordy HTML artifact without
    # section structure stays abstract_only no matter how long it is.
    extraction = ScholarExtraction(
        markdown="# Title\n\n" + "word " * 1_000,
        section_titles=(),
        reference_count=0,
        references_resolved=False,
        word_count=1_002,
    )
    paper = AcquiredPaper(
        "html",
        b"<html><body>word " * 1,
        3,
        "arxiv-html",
        "https://arxiv.org/html/1706.03762",
        200,
        ("https://arxiv.org/html/1706.03762",),
        True,
    )

    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(
            title="Deep learning",
            abstract="A summary of the work.",
            first_page="436",
            last_page="444",
        ),
        acquisition=paper,
    )

    assert verdict.content_level == "abstract_only"
    assert verdict.completeness.status == "incomplete"
