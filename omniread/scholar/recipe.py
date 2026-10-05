"""Recipe orchestration for scholarly artifact-level reads."""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import urlsplit

from ..recipes.base import (
    HttpResponse,
    ReadOptions,
    Recipe,
    RecipeContext,
    clock_iso,
    shape_recipe_result,
)
from ..types import (
    Completeness,
    DependencyError,
    CostToComplete,
    Evidence,
    JsonValue,
    OmniReadError,
    Provenance,
    ReadResult,
    RetrievalError,
)
from .extract import (
    ScholarExtraction,
    extract_abstract_from_html,
    extract_paper,
    extract_title_from_html,
)
from .ids import (
    SCHOLARLY_HOSTS,
    ScholarlyIdentifier,
    is_scholarly_input,
    normalize_identifier,
)
from .http import get_with_retry
from .identity import verify_identity
from .locate import AcquiredPaper, AcquisitionResult, acquire_full_text
from .metadata import ScholarMetadata, collect_metadata
from .verify import ScholarVerdict, verify_scholarly


class ScholarRecipe(Recipe):
    """Acquire the paper artifact, never merely certify its landing page."""

    name = "academic"
    priority = 90
    RECIPE_META = {
        "name": name,
        "domains": sorted(SCHOLARLY_HOSTS),
        "maturity": "experimental",
        "license_note": (
            "Trafilatura and Courlan are Apache-2.0; pypdf is BSD-3-Clause; "
            "selectolax and optional Docling OCR are MIT"
        ),
        "test_urls": [
            "https://doi.org/10.1371/journal.pmed.0020124",
            "https://arxiv.org/abs/1706.03762",
            "https://www.nature.com/articles/nature14539",
        ],
    }

    @classmethod
    def match(cls, url: str) -> bool:
        """Own supported identifiers and the broad scholarly domain set."""

        return is_scholarly_input(url)

    @staticmethod
    def parse_target(value: str) -> tuple[str, str]:
        """Compatibility parser returning the canonical kind/value pair."""

        identifier = normalize_identifier(value)
        return identifier.kind, identifier.value

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        landing, landing_error = _landing_page(url, ctx)
        try:
            identifier = normalize_identifier(
                url,
                page_html=landing.text if landing is not None else None,
            )
        except ValueError as exc:
            return _unidentified_result(
                url=url,
                landing=landing,
                landing_error=landing_error,
                identifier_error=str(exc),
                opts=opts,
                ctx=ctx,
            )
        metadata = collect_metadata(identifier, ctx.http)
        _seed_identity(metadata, identifier)
        fallback_abstract = (
            extract_abstract_from_html(landing.text) if landing is not None else None
        )
        if metadata.abstract is None:
            metadata.abstract = fallback_abstract
        if metadata.title is None and landing is not None:
            metadata.title = extract_title_from_html(landing.text)

        selected_extraction: ScholarExtraction | None = None
        extraction_failures: list[str] = []

        def validate_candidate(paper: AcquiredPaper) -> tuple[bool, str]:
            nonlocal selected_extraction
            try:
                identity_ok, identity_detail = verify_identity(
                    paper.payload,
                    paper.kind,
                    metadata,
                    bound=paper.identity_bound,
                )
            except DependencyError as exc:
                extraction_failures.append(str(exc))
                return False, str(exc)
            if not identity_ok:
                extraction_failures.append(identity_detail)
                return False, identity_detail
            try:
                candidate = extract_paper(paper)
            except OmniReadError as exc:
                extraction_failures.append(str(exc))
                return False, str(exc)
            except Exception as exc:
                if paper.kind != "pdf":
                    raise
                detail = f"PDF conversion failed: {exc}"
                extraction_failures.append(detail)
                return False, detail
            candidate_verdict = verify_scholarly(
                extraction=candidate,
                metadata=metadata,
                acquisition=paper,
                fallback_abstract=fallback_abstract,
            )
            if candidate_verdict.content_level in {"metadata_only", "abstract_only"}:
                detail = (
                    "The fetched representation did not contain an independently "
                    "demonstrated paper body"
                )
                extraction_failures.append(detail)
                return False, detail
            selected_extraction = candidate
            return (
                True,
                f"{identity_detail}; representation extracted into structured scholarly Markdown",
            )

        acquisition = acquire_full_text(
            identifier=identifier,
            metadata=metadata,
            original_url=_absolute_input(url, identifier),
            landing=landing,
            opts=opts,
            ctx=ctx,
            validate=validate_candidate,
        )
        extraction = selected_extraction
        extraction_error = (
            extraction_failures[-1]
            if acquisition.paper is None and extraction_failures
            else None
        )

        verdict = verify_scholarly(
            extraction=extraction,
            metadata=metadata,
            acquisition=acquisition.paper,
            fallback_abstract=fallback_abstract,
        )
        fallback_content: str | None = None
        fallback_result: ReadResult | None = None
        fallback_error: str | None = None
        if acquisition.paper is None:
            fallback_content, fallback_result, fallback_error = _page_fallback(
                url=url,
                landing=landing,
                ctx=ctx,
            )
        unlock_path = _unlock_path(
            acquisition=acquisition,
            extraction_error=extraction_error,
        )
        if verdict.completeness.status == "incomplete":
            verdict = replace(
                verdict,
                completeness=replace(
                    verdict.completeness,
                    reason=f"{verdict.completeness.reason} Unlock path: {unlock_path}.",
                ),
            )
        markdown = (
            extraction.markdown
            if extraction is not None and acquisition.paper is not None
            else fallback_content or _partial_markdown(metadata)
        )
        cost = (
            _completion_cost(metadata, unlock_path)
            if verdict.completeness.status == "incomplete"
            else None
        )
        structured = _structured_data(
            identifier=identifier,
            metadata=metadata,
            verdict=verdict,
            extraction=extraction if acquisition.paper is not None else None,
            acquisition=acquisition,
            unlock_path=unlock_path,
            landing_error=landing_error,
            extraction_error=extraction_error,
            fallback_error=fallback_error,
        )
        provenance = _provenance(
            url=url,
            identifier=identifier,
            metadata=metadata,
            landing=landing,
            acquisition=acquisition,
            ctx=ctx,
            fallback_result=fallback_result,
        )
        return shape_recipe_result(
            url=url,
            markdown=markdown,
            completeness=verdict.completeness,
            provenance=provenance,
            structured_data=structured,
            opts=opts,
            cost_to_complete=cost,
        )


def _landing_page(
    value: str,
    ctx: RecipeContext,
) -> tuple[HttpResponse | None, str | None]:
    try:
        identifier = normalize_identifier(value)
    except ValueError:
        target = value
    else:
        target = _absolute_input(value, identifier)
    try:
        return get_with_retry(ctx.http, target), None
    except RetrievalError as exc:
        return None, str(exc)


def _unidentified_result(
    *,
    url: str,
    landing: HttpResponse | None,
    landing_error: str | None,
    identifier_error: str,
    opts: ReadOptions,
    ctx: RecipeContext,
) -> ReadResult:
    """Return an honest metadata-only page result when no work ID can be resolved."""

    fallback_content, fallback_result, fallback_error = _page_fallback(
        url=url,
        landing=landing,
        ctx=ctx,
    )
    title = (
        extract_title_from_html(landing.text)
        if landing is not None
        else None
    ) or "Unidentified scholarly page"
    markdown = fallback_content or (
        f"# {title}\n\nNo canonical DOI, arXiv ID, PMID, or PMCID could be "
        "resolved for this page."
    )
    unlock_path = "provide a canonical DOI, arXiv ID, PMID, or PMCID and retry"
    completeness = Completeness(
        "incomplete",
        [
            Evidence(
                "scholarly_identifier",
                False,
                identifier_error,
            ),
            Evidence(
                "landing_fetch",
                landing is not None,
                (
                    f"Fetched scholarly page with HTTP {landing.status}"
                    if landing is not None
                    else f"Scholarly page fetch failed: {landing_error or 'unknown error'}"
                ),
            ),
        ],
        "The page could not be tied to a canonical scholarly work, so only "
        f"page-level metadata is available. Unlock path: {unlock_path}.",
    )
    metadata = ScholarMetadata(title=title)
    final_url = landing.final_url if landing is not None else url
    if fallback_result is not None:
        provenance = replace(
            fallback_result.provenance,
            recipe="academic",
            canonical_url=url,
            source_urls=list(
                dict.fromkeys(
                    (
                        url,
                        *fallback_result.provenance.source_urls,
                        *([landing.final_url] if landing is not None else []),
                    )
                )
            ),
        )
    else:
        provenance = Provenance(
            tier=0 if landing is not None else None,
            engine=(
                "trafilatura-scholarly-fallback"
                if fallback_content is not None
                else "scholarly-page-identity"
            ),
            recipe="academic",
            canonical_url=url,
            fetched_at=clock_iso(ctx.clock),
            http_status=landing.status if landing is not None else None,
            final_url=final_url,
            source_urls=list(
                dict.fromkeys((url, *([landing.final_url] if landing is not None else [])))
            ),
        )
    structured: JsonValue = {
        "scholarly": {
            "content_level": "metadata_only",
            "identifier": None,
            "bibliographic": metadata.as_json(),
            "full_text_acquisition": None,
            "section_titles": [],
            "found_reference_count": 0,
            "acquisition_attempts": [],
            "unlock_path": unlock_path,
            "landing_error": landing_error,
            "extraction_error": identifier_error,
            "page_fallback_error": fallback_error,
        }
    }
    return shape_recipe_result(
        url=url,
        markdown=markdown,
        completeness=completeness,
        provenance=provenance,
        structured_data=structured,
        opts=opts,
        cost_to_complete=CostToComplete(
            remaining_items=1,
            estimated_extra_tokens=1_000,
        ),
    )


def _absolute_input(value: str, identifier: ScholarlyIdentifier) -> str:
    try:
        parsed = urlsplit(value)
    except (TypeError, ValueError):
        return identifier.canonical_url
    return value if parsed.scheme in {"http", "https"} else identifier.canonical_url


def _page_fallback(
    *,
    url: str,
    landing: HttpResponse | None,
    ctx: RecipeContext,
) -> tuple[str | None, ReadResult | None, str | None]:
    """Keep readable page content while the scholarly verdict remains fail-closed."""

    fallback_error: str | None = None
    try:
        result = ctx.generic_read(
            url,
            budget=None,
            full=True,
            clock=ctx.clock,
            render=ctx.render,
        )
        if result.content.strip():
            return result.content, result, None
        fallback_error = "The generic ladder returned no readable page content"
    except Exception as exc:
        # This boundary exists specifically so a failed fallback cannot restore the
        # exception behavior ADR 0006 forbids. The primary scholarly verdict still
        # records the acquisition failure.
        fallback_error = f"Generic page fallback failed: {exc}"
    if landing is not None:
        try:
            extracted = extract_paper(
                AcquiredPaper(
                    kind="html",
                    payload=landing.body,
                    rung=10,
                    provider="landing-page-fallback",
                    final_url=landing.final_url,
                    http_status=landing.status,
                    source_urls=(url, landing.final_url),
                )
            )
        except Exception as exc:
            detail = f"Landing-page fallback extraction failed: {exc}"
            fallback_error = (
                f"{fallback_error}; {detail}" if fallback_error else detail
            )
        else:
            return extracted.markdown, None, fallback_error
    return None, None, fallback_error


def _seed_identity(
    metadata: ScholarMetadata,
    identifier: ScholarlyIdentifier,
) -> None:
    if identifier.kind == "doi" and metadata.doi is None:
        metadata.doi = identifier.value
    elif identifier.kind == "arxiv" and metadata.arxiv_id is None:
        metadata.arxiv_id = identifier.value
    elif identifier.kind == "pmid" and metadata.pmid is None:
        metadata.pmid = identifier.value
    elif identifier.kind == "pmcid" and metadata.pmcid is None:
        metadata.pmcid = identifier.value


def _partial_markdown(metadata: ScholarMetadata) -> str:
    title = metadata.title or "Scholarly work"
    parts = [f"# {title}"]
    if metadata.abstract:
        parts.append(f"## Abstract\n\n{metadata.abstract}")
    else:
        identifiers = [
            f"DOI: {metadata.doi}" if metadata.doi else None,
            f"PMID: {metadata.pmid}" if metadata.pmid else None,
            f"PMCID: {metadata.pmcid}" if metadata.pmcid else None,
            f"arXiv: {metadata.arxiv_id}" if metadata.arxiv_id else None,
        ]
        parts.append("\n".join(value for value in identifiers if value) or "Metadata only.")
    return "\n\n".join(parts)


def _unlock_path(
    *,
    acquisition: AcquisitionResult,
    extraction_error: str | None,
) -> str:
    if extraction_error and "[pdf]" in extraction_error:
        return "install OmniRead's pdf extra (pypdf) and retry"
    if extraction_error and "Docling" in extraction_error:
        return "install OmniRead's MIT-licensed academic extra (Docling) and retry"
    if acquisition.untried_unlocks:
        return acquisition.untried_unlocks[0]
    return "obtain a reachable OA copy or refresh the stored Tier-4 publisher profile"


def _completion_cost(
    metadata: ScholarMetadata,
    unlock_path: str,
) -> CostToComplete:
    pages = _page_count(metadata.first_page, metadata.last_page)
    if pages is not None:
        estimated_tokens = pages * 450
    elif metadata.abstract:
        estimated_tokens = max(1_000, len(metadata.abstract.split()) * 6)
    else:
        estimated_tokens = 1_000
    required_tier = (
        4 if "omniread login" in unlock_path or "Tier-4" in unlock_path else None
    )
    return CostToComplete(
        remaining_items=1,
        estimated_extra_tokens=estimated_tokens,
        required_tier=required_tier,
    )


def _page_count(first: str | None, last: str | None) -> int | None:
    if not first or not last or not first.isdigit() or not last.isdigit():
        return None
    start, end = int(first), int(last)
    return end - start + 1 if end >= start else None


def _structured_data(
    *,
    identifier: ScholarlyIdentifier,
    metadata: ScholarMetadata,
    verdict: ScholarVerdict,
    extraction: ScholarExtraction | None,
    acquisition: AcquisitionResult,
    unlock_path: str,
    landing_error: str | None,
    extraction_error: str | None,
    fallback_error: str | None,
) -> JsonValue:
    return {
        "scholarly": {
            "content_level": verdict.content_level,
            "identifier": {
                "kind": identifier.kind,
                "value": identifier.value,
            },
            "bibliographic": metadata.as_json(),
            "full_text_acquisition": (
                {
                    "rung": acquisition.paper.rung,
                    "provider": acquisition.paper.provider,
                    "kind": acquisition.paper.kind,
                }
                if acquisition.paper is not None
                else None
            ),
            "last_resort": (
                {
                    "pdf_path": acquisition.last_resort_pdf_path,
                    "provider": (
                        acquisition.paper.provider
                        if acquisition.paper is not None
                        else None
                    ),
                }
                if acquisition.last_resort_pdf_path is not None
                else None
            ),
            "section_titles": list(extraction.section_titles) if extraction else [],
            "found_reference_count": extraction.reference_count if extraction else 0,
            "acquisition_attempts": [item.as_json() for item in acquisition.attempts],
            "unlock_path": unlock_path,
            "landing_error": landing_error,
            "extraction_error": extraction_error,
            "page_fallback_error": fallback_error,
        }
    }


def _provenance(
    *,
    url: str,
    identifier: ScholarlyIdentifier,
    metadata: ScholarMetadata,
    landing: HttpResponse | None,
    acquisition: AcquisitionResult,
    ctx: RecipeContext,
    fallback_result: ReadResult | None,
) -> Provenance:
    paper = acquisition.paper
    tier = None
    engine = "scholarly-metadata-manifests"
    final_url = landing.final_url if landing else identifier.canonical_url
    status = landing.status if landing else None
    if paper is not None:
        tier = paper.tier if paper.tier is not None else 2 if paper.kind == "pdf" else 0
        engine = paper.provider
        final_url = paper.final_url
        status = paper.http_status
    elif fallback_result is not None:
        tier = fallback_result.provenance.tier
        engine = fallback_result.provenance.engine
        final_url = fallback_result.provenance.final_url
        status = fallback_result.provenance.http_status
    sources = [
        url,
        *(metadata.source_urls),
        *([landing.final_url] if landing else []),
        *(acquisition.source_urls),
        *(fallback_result.provenance.source_urls if fallback_result else []),
    ]
    return Provenance(
        tier=tier,
        engine=engine,
        recipe="academic",
        canonical_url=identifier.canonical_url,
        fetched_at=clock_iso(ctx.clock),
        http_status=status,
        final_url=final_url,
        source_urls=list(dict.fromkeys(sources)),
    )
