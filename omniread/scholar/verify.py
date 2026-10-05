"""Artifact-level completeness verification for scholarly reads."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, TypeAlias

from ..types import Completeness, Evidence
from .extract import ScholarExtraction
from .locate import AcquiredPaper
from .metadata import ScholarMetadata

ContentLevel: TypeAlias = Literal[
    "metadata_only",
    "abstract_only",
    "full_text",
    "full_text_with_references",
]

_NON_BODY_SECTIONS = {
    "abstract",
    "access options",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "author information",
    "authors and affiliations",
    "corresponding author",
    "ethics declarations",
    "competing interests",
    "additional information",
    "rights and permissions",
    "about this article",
    "cite this article",
    "comments",
    "supplementary information",
    "editorial summary",
    "similar content being viewed by others",
}
_BACK_MATTER_STARTS = (
    "references",
    "bibliography",
    "acknowledgement",
    "acknowledgment",
    "appendix",
    "appendices",
    "supplementary information",
    "author information",
    "ethics declaration",
    "competing interest",
    "rights and permissions",
    "about this article",
    "cite this article",
)
_MIN_BODY_WORDS = 500
_MIN_UNMANIFESTED_REFERENCES = 5


@dataclass(frozen=True, slots=True)
class ScholarVerdict:
    """The paper kind held and the shared tri-state honesty verdict."""

    content_level: ContentLevel
    completeness: Completeness


def verify_scholarly(
    *,
    extraction: ScholarExtraction | None,
    metadata: ScholarMetadata,
    acquisition: AcquiredPaper | None,
    fallback_abstract: str | None = None,
) -> ScholarVerdict:
    """Judge the artifact from independent acquisition and manifest signals."""

    abstract = metadata.abstract or fallback_abstract
    kind = acquisition.kind if acquisition is not None else None
    acquisition_evidence = Evidence(
        "full_text_acquisition",
        acquisition is not None and extraction is not None,
        (
            f"Rung {acquisition.rung} ({acquisition.provider}) returned classified "
            f"{acquisition.kind} content"
            if acquisition is not None and extraction is not None
            else "No acquisition rung returned a classified scholarly body"
        ),
    )
    section_evidence = _section_manifest(extraction, abstract, kind)
    abstract_evidence = _abstract_containment(extraction, abstract, kind)
    extent_evidence = _page_extent(extraction, metadata, kind)
    reference_evidence = _reference_manifest(extraction, metadata)
    evidence = [
        acquisition_evidence,
        section_evidence,
        abstract_evidence,
        extent_evidence,
        reference_evidence,
    ]

    # A representation without a proved body is an abstract/metadata artifact even
    # when it came from an authenticated page or carried a long reference sidebar.
    # A PDF is a structural source (ADR 0008): when its section headings were not
    # observable in the text layer, a page extent that corroborates the extracted
    # word count is the independent body proof.
    has_body = (
        acquisition is not None
        and extraction is not None
        and abstract_evidence.passed is not False
        and (
            section_evidence.passed is True
            or (kind == "pdf" and extent_evidence.passed is True)
        )
    )
    if not has_body:
        level: ContentLevel = (
            "abstract_only"
            if abstract or (extraction is not None and _contains_abstract_heading(extraction))
            else "metadata_only"
        )
        return ScholarVerdict(
            level,
            Completeness(
                "incomplete",
                evidence,
                "Only scholarly metadata or an abstract was obtained; the paper body "
                "was not independently demonstrated.",
            ),
        )

    level = (
        "full_text_with_references"
        if extraction.references_resolved and extraction.reference_count > 0
        else "full_text"
    )
    # PDF text layers hide or merge section headings and reference entries
    # (accepted-article and scanned templates), so for PDF artifacts the section
    # and reference manifests are extraction measurements, not content-gap
    # signals. They still gate the "complete" level below. The amount vetoes
    # remain: a body not longer than its abstract or short of its page range is
    # a genuine gap regardless of how the text layer was laid out. A reference
    # manifest that reports literally zero entries against two positive provider
    # manifests is still a veto: a declared 100-reference paper with nothing
    # observable is a crafted or partial artifact, not a counting artifact.
    if kind == "pdf":
        checks = [abstract_evidence, extent_evidence]
        if (
            reference_evidence.passed is False
            and extraction is not None
            and extraction.reference_count == 0
        ):
            checks.append(reference_evidence)
    else:
        checks = [
            section_evidence,
            abstract_evidence,
            extent_evidence,
            reference_evidence,
        ]
    body_failures = [item for item in checks if item.passed is False]
    if body_failures:
        return ScholarVerdict(
            level,
            Completeness(
                "incomplete",
                evidence,
                "Independent body evidence identifies a scholarly content gap.",
            ),
        )

    # An available page manifest with inconclusive extent cannot be overruled
    # by sections and references: an excerpt can preserve both. Short or
    # figure-heavy papers remain unknown rather than being called incomplete.
    if (
        extent_evidence.passed is None
        and _numeric_page_count(metadata.first_page, metadata.last_page) is not None
    ):
        return ScholarVerdict(
            level,
            Completeness(
                "unknown",
                evidence,
                "The body extent is inconclusive against the declared page range; "
                "sections and references cannot certify an excerpt as complete.",
            ),
        )

    # Section structure proves that a body exists; a substantial reference list is
    # the independent manifest that can certify it. Page/abstract extents remain
    # asymmetric vetoes because they share the extracted word count with section
    # evidence and therefore cannot be counted as independent votes.
    if section_evidence.passed is True and reference_evidence.passed is True:
        return ScholarVerdict(
            level,
            Completeness(
                "complete",
                evidence,
                "Independent body structure and reference manifests support the "
                "acquired paper.",
            ),
        )
    return ScholarVerdict(
        level,
        Completeness(
            "unknown",
            evidence,
            "A body was acquired, but the independent paper manifests are too thin "
            "to certify completeness.",
        ),
    )


def _section_manifest(
    extraction: ScholarExtraction | None,
    abstract: str | None,
    kind: str | None = None,
) -> Evidence:
    if extraction is None:
        return Evidence(
            "section_manifest",
            False,
            "No extracted representation exists from which to observe body sections",
        )
    substantive, body_words = _body_profile(extraction)
    enough_sections = len(substantive) >= 2
    clear_margin = body_words >= _MIN_BODY_WORDS
    margin_detail = f"; observed {body_words} words inside the main body"
    if abstract:
        abstract_words = max(1, len(abstract.split()))
        ratio = body_words / abstract_words
        clear_margin = clear_margin and body_words >= abstract_words + 300 and ratio >= 2.5
        margin_detail = (
            f"; observed {body_words} main-body words against a "
            f"{abstract_words}-word abstract"
        )
    passed = enough_sections and clear_margin
    return Evidence(
        "section_manifest",
        passed,
        (
            f"Observed {len(substantive)} substantive sections: "
            + ", ".join(substantive[:6])
            + margin_detail
            if passed
            else (
                f"Body proof requires at least two substantive pre-back-matter "
                f"sections and {_MIN_BODY_WORDS} main-body words with a clear "
                f"margin beyond the abstract; observed {len(substantive)}"
                + margin_detail
            )
        ),
    )


def _abstract_containment(
    extraction: ScholarExtraction | None,
    abstract: str | None,
    kind: str | None = None,
) -> Evidence:
    if extraction is None:
        return Evidence(
            "abstract_containment",
            None,
            "No acquired representation exists for abstract-length comparison",
        )
    if not abstract:
        return Evidence(
            "abstract_containment",
            None,
            "No independent abstract was available for length comparison",
        )
    abstract_words = max(1, len(abstract.split()))
    body_words = _body_word_count(extraction, kind)
    ratio = body_words / abstract_words
    passed: bool | None
    if body_words <= abstract_words + 100 or ratio <= 1.5:
        passed = False
    else:
        passed = None
    return Evidence(
        "abstract_containment",
        passed,
        f"Extracted {body_words} words against an independent {abstract_words}-word abstract",
    )


def _page_extent(
    extraction: ScholarExtraction | None,
    metadata: ScholarMetadata,
    kind: str | None = None,
) -> Evidence:
    if extraction is None:
        return Evidence(
            "page_extent",
            None,
            "No acquired representation exists for page-extent comparison",
        )
    pages = _numeric_page_count(metadata.first_page, metadata.last_page)
    if pages is None:
        return Evidence(
            "page_extent",
            None,
            "No numeric first/last-page manifest was available",
        )
    expected_words = pages * 450
    body_words = _body_word_count(extraction, kind)
    if body_words >= expected_words * 0.50:
        passed: bool | None = True
    elif body_words < expected_words * 0.20:
        passed = False
    else:
        passed = None
    return Evidence(
        "page_extent",
        passed,
        f"Observed {body_words} main-body words against {expected_words} expected "
        f"for a {pages}-page manifest",
    )


def _body_word_count(extraction: ScholarExtraction, kind: str | None) -> int:
    """Measure the main body for the body-evidence checks.

    PDF artifacts are counted by their whole extracted text: section headings are
    not always recoverable from the text layer (accepted-article and scanned
    templates), so a heading-based count would understate or hide a complete body.
    """

    if kind == "pdf":
        return extraction.word_count
    _, body_words = _body_profile(extraction)
    return body_words


def _reference_manifest(
    extraction: ScholarExtraction | None,
    metadata: ScholarMetadata,
) -> Evidence:
    counts = (
        metadata.crossref_reference_count,
        metadata.openalex_reference_count,
    )
    positive = [value for value in counts if isinstance(value, int) and value > 0]
    found = extraction.reference_count if extraction is not None else 0
    if len(positive) < 2:
        passed = True if found >= _MIN_UNMANIFESTED_REFERENCES else None
        return Evidence(
            "reference_manifest",
            passed,
            (
                f"Observed a substantial {found}-item reference list"
                if passed is True
                else (
                    f"Observed only {found} references, fewer than the "
                    f"{_MIN_UNMANIFESTED_REFERENCES} required without two provider "
                    "manifests"
                    if found > 0
                    else "No reference list was observed and both positive Crossref "
                    "and OpenAlex counts are unavailable"
                )
            ),
        )
    lower = min(positive)
    if found < lower * 0.40:
        passed: bool | None = False
    elif found >= lower * 0.70:
        passed = True
    else:
        passed = None
    return Evidence(
        "reference_manifest",
        passed,
        f"Found {found} references; Crossref declares {positive[0]} and "
        f"OpenAlex declares {positive[1]}",
    )


def _numeric_page_count(
    first_page: str | None,
    last_page: str | None,
) -> int | None:
    if not first_page or not last_page:
        return None
    first = re.fullmatch(r"\d+", first_page.strip())
    last = re.fullmatch(r"\d+", last_page.strip())
    if first is None or last is None:
        return None
    start, end = int(first.group()), int(last.group())
    return end - start + 1 if end >= start else None


def _is_substantive_section(title: str) -> bool:
    normalized = _normalized_heading(title)
    if not normalized or normalized in _NON_BODY_SECTIONS:
        return False
    return not normalized.startswith(
        (
            "access ",
            "author ",
            "similar content",
            "rights ",
            "ethics ",
            "competing ",
            "about this ",
            "cite this ",
        )
    )


def _contains_abstract_heading(extraction: ScholarExtraction) -> bool:
    return any(title.strip().casefold() == "abstract" for title in extraction.section_titles)


def _body_profile(extraction: ScholarExtraction) -> tuple[list[str], int]:
    """Measure prose structurally between the abstract and scholarly back matter."""

    headings = list(
        re.finditer(r"(?m)^#{2,6}\s+(.+?)\s*$", extraction.markdown)
    )
    substantive: list[str] = []
    body_words = 0
    for index, match in enumerate(headings):
        title = match.group(1).strip()
        normalized = _normalized_heading(title)
        if any(normalized.startswith(prefix) for prefix in _BACK_MATTER_STARTS):
            break
        if not _is_substantive_section(title):
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(
            extraction.markdown
        )
        section_words = len(extraction.markdown[match.end() : end].split())
        if section_words == 0:
            continue
        substantive.append(title)
        body_words += section_words
    return substantive, body_words


def _normalized_heading(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\s*", "", title).strip().casefold()
