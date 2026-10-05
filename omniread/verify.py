"""Independent completeness verification for extracted web content."""

from __future__ import annotations

import math
import re

from ._html import (
    MD_FENCE_RE,
    MD_HEADING_RE,
    declared_word_count,
    has_pagination,
    normalize_space,
    visible_dom_snapshot,
    word_count,
)
from .extract import ExtractedPage
from .fetch import FetchedPage
from .types import Completeness, Evidence
_HEADING_COMPLETE_RATIO = 0.90
_HEADING_INCOMPLETE_RATIO = 0.60


def verify_completeness(
    fetch: FetchedPage,
    extracted: ExtractedPage,
    *,
    pagination_covered: bool = False,
) -> Completeness:
    """Decide completeness from evidence independent of Trafilatura's confidence.

    A positive block-page signal always fails closed as ``unknown``. A genuine
    coverage gap is ``incomplete``. ``complete`` requires at least two independent
    positive content signals, so plausible prose alone can never earn that verdict.
    """

    snapshot = visible_dom_snapshot(fetch.raw_html)
    output_words = word_count(extracted.markdown)
    output_headings = _markdown_headings(extracted.markdown)

    block_evidence = Evidence(
        name="blockpage_check",
        passed=not fetch.block_signal.detected,
        detail=(
            "No known block or consent-wall signal detected"
            if not fetch.block_signal.detected
            else "; ".join(fetch.block_signal.reasons)
        ),
    )
    paginated = has_pagination(fetch.raw_html, current_url=fetch.final_url)
    if paginated:
        pagination_evidence = Evidence(
            name="pagination_coverage",
            passed=True if pagination_covered else False,
            detail=(
                "The recipe followed and independently verified every advertised page"
                if pagination_covered
                else "The source advertises pagination but only one page is present"
            ),
        )
    else:
        pagination_evidence = Evidence(
            name="pagination_coverage",
            passed=None,
            detail="The fetched representation exposes no explicit pagination controls",
        )

    if not snapshot.headings:
        heading_evidence = Evidence(
            name="heading_presence",
            passed=None,
            detail="The visible DOM exposes no headings to use as an independent manifest",
        )
    else:
        # Engines may emit headings as plain or bold standalone lines. A word
        # occurring inside a paragraph does not demonstrate the named section.
        normalized_output = {_normalize_heading(value) for value in output_headings}
        standalone_lines = {_normalize_heading(line) for line in extracted.markdown.splitlines()}

        def _heading_represented(heading: str) -> bool:
            norm = _normalize_heading(heading)
            if not norm:
                return True
            return norm in normalized_output or norm in standalone_lines

        missing = [
            heading for heading in snapshot.headings if not _heading_represented(heading)
        ]
        represented = len(snapshot.headings) - len(missing)
        represented_ratio = represented / len(snapshot.headings)
        if represented_ratio >= _HEADING_COMPLETE_RATIO:
            heading_passed: bool | None = True
            interpretation = "meets the strong-agreement floor"
        elif represented_ratio <= _HEADING_INCOMPLETE_RATIO:
            heading_passed = False
            interpretation = "falls below the large-gap ceiling"
        else:
            heading_passed = None
            interpretation = "falls in the verifier's indeterminate band"
        missing_examples = ", ".join(missing[:3])
        if len(missing) > 3:
            missing_examples += f" (+{len(missing) - 3} more)"
        heading_evidence = Evidence(
            name="heading_presence",
            passed=heading_passed,
            detail=(
                f"{represented} of {len(snapshot.headings)} visible DOM headings are "
                f"represented as Markdown headings (ratio {represented_ratio:.2f}) and "
                f"{interpretation}"
                + (f"; missing examples: {missing_examples}" if missing else "")
            ),
        )

    dom_words = snapshot.word_count
    if dom_words < 100:
        coverage_evidence = Evidence(
            name="dom_vs_output_coverage",
            passed=None,
            detail=(
                f"Visible main DOM has only {dom_words} words; that reference is too "
                "small for a reliable coverage decision"
            ),
        )
    else:
        complete_floor = math.ceil(dom_words * 0.80)
        incomplete_ceiling = math.floor(dom_words * 0.60)
        if output_words >= complete_floor:
            passed: bool | None = True
            interpretation = "meets the strong-coverage floor"
        elif output_words <= incomplete_ceiling:
            passed = False
            interpretation = "falls below the large-gap ceiling"
        else:
            passed = None
            interpretation = "falls in the verifier's indeterminate band"
        coverage_evidence = Evidence(
            name="dom_vs_output_coverage",
            passed=passed,
            detail=(
                f"Extracted Markdown has {output_words} words versus {dom_words} visible "
                f"main-DOM words and {interpretation}"
            ),
        )

    declared = declared_word_count(fetch.raw_html, extracted.structured_data)
    if declared is None:
        length_evidence = Evidence(
            name="length_sanity",
            passed=None,
            detail="No JSON-LD or metadata word-count declaration is available",
        )
    else:
        complete_floor = math.ceil(declared * 0.85)
        incomplete_ceiling = math.floor(declared * 0.70)
        if output_words >= complete_floor:
            length_passed: bool | None = True
            interpretation = "is consistent with the declared length"
        elif output_words <= incomplete_ceiling:
            length_passed = False
            interpretation = "is materially below the declared length"
        else:
            length_passed = None
            interpretation = "is close enough to be ambiguous but not enough to certify"
        length_evidence = Evidence(
            name="length_sanity",
            passed=length_passed,
            detail=(
                f"Extracted Markdown has {output_words} words; the publisher declares "
                f"{declared}, so the extracted length {interpretation}"
            ),
        )

    evidence = [
        heading_evidence,
        coverage_evidence,
        block_evidence,
        length_evidence,
        pagination_evidence,
    ]

    if fetch.block_signal.detected:
        return Completeness(
            status="unknown",
            evidence=evidence,
            reason="A block/challenge representation may have replaced the requested page",
        )

    if pagination_evidence.passed is False:
        return Completeness(
            status="incomplete",
            evidence=evidence,
            reason="Independent pagination evidence shows that later pages were not retrieved",
        )

    content_signals = [heading_evidence, coverage_evidence, length_evidence]
    failed = [signal.name for signal in content_signals if signal.passed is False]
    if failed:
        return Completeness(
            status="incomplete",
            evidence=evidence,
            reason=f"Independent evidence identifies a content gap: {', '.join(failed)}",
        )

    passed = [signal.name for signal in content_signals if signal.passed is True]
    if len(passed) >= 2:
        return Completeness(
            status="complete",
            evidence=evidence,
            reason=f"Independent coverage signals agree: {', '.join(passed)}",
        )

    return Completeness(
        status="unknown",
        evidence=evidence,
        reason="The available independent evidence is too thin to certify completeness",
    )


def _markdown_headings(markdown: str) -> tuple[str, ...]:
    headings: list[str] = []
    active_fence: str | None = None
    for line in markdown.splitlines():
        fence = MD_FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            marker_char = marker[0]
            if active_fence is None:
                active_fence = marker_char
            elif active_fence == marker_char:
                active_fence = None
            continue
        if active_fence is not None:
            continue
        match = MD_HEADING_RE.match(line)
        if match:
            headings.append(normalize_space(match.group(2)))
    return tuple(headings)


def _normalize_heading(value: str) -> str:
    value = re.sub(r"[`*_~\[\]()]", "", value)
    value = re.sub(r"[^\w]+", " ", value.lower(), flags=re.UNICODE)
    return normalize_space(value)
