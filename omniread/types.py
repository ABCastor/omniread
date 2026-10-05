"""The stable retrieval contract shared by OmniRead's core and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias, get_args
from urllib.parse import urlsplit, urlunsplit

CompletenessStatus: TypeAlias = Literal["complete", "incomplete", "unknown"]
Coverage: TypeAlias = Literal["core", "full"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """One independently observed verifier signal.

    ``passed`` is ``None`` when the signal cannot make a reliable judgment.
    ``detail`` contains measurements or facts, never an ungrounded confidence score.
    """

    name: str
    passed: bool | None
    detail: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Evidence.name must not be empty")
        if not self.detail.strip():
            raise ValueError("Evidence.detail must not be empty")


@dataclass(frozen=True, slots=True)
class Completeness:
    """The verifier's tri-state decision and the evidence supporting it."""

    status: CompletenessStatus
    evidence: list[Evidence]
    reason: str

    def __post_init__(self) -> None:
        if self.status not in get_args(CompletenessStatus):
            raise ValueError(f"Invalid completeness status: {self.status!r}")
        if not self.reason.strip():
            raise ValueError("Completeness.reason must not be empty")


@dataclass(frozen=True, slots=True)
class Section:
    """An ordered Markdown section; ``level`` expresses its tree depth."""

    heading: str
    level: int
    anchor: str
    token_count: int
    char_count: int

    def __post_init__(self) -> None:
        if not self.heading.strip() or not self.anchor.strip():
            raise ValueError("Section heading and anchor must not be empty")
        if not 0 <= self.level <= 6:
            raise ValueError("Section.level must be between 0 and 6")
        if self.token_count < 0 or self.char_count < 0:
            raise ValueError("Section counts must not be negative")


@dataclass(frozen=True, slots=True)
class Outline:
    """The full document map, with approximate token counts."""

    sections: list[Section]
    total_token_count: int
    has_tables: bool
    has_code: bool
    has_pagination: bool

    def __post_init__(self) -> None:
        if self.total_token_count < 0:
            raise ValueError("Outline.total_token_count must not be negative")


@dataclass(frozen=True, slots=True)
class CostToComplete:
    """Measured work remaining after a bounded response or known retrieval gap.

    The item count and token estimate are always present. Browser tier and latency
    are optional because they are only meaningful when completing the response
    requires another retrieval rung.
    """

    remaining_items: int
    estimated_extra_tokens: int
    required_tier: int | None = None
    estimated_latency_ms: int | None = None

    def __post_init__(self) -> None:
        measurements = (self.remaining_items, self.estimated_extra_tokens)
        if any(value < 0 for value in measurements):
            raise ValueError("Cost-to-complete measurements must be non-negative")
        if self.required_tier is not None and self.required_tier < 0:
            raise ValueError("Cost-to-complete tier must be non-negative")
        if self.estimated_latency_ms is not None and self.estimated_latency_ms < 0:
            raise ValueError("Cost-to-complete latency must be non-negative")


@dataclass(frozen=True, slots=True)
class Provenance:
    """How and when a result was retrieved.

    The calling boundary supplies ``fetched_at``. Core retrieval code never reads a
    hidden system clock.
    """

    tier: int | None
    engine: str
    recipe: str
    canonical_url: str
    fetched_at: str
    http_status: int | None
    final_url: str
    source_urls: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_url", _public_provenance_url(self.canonical_url))
        object.__setattr__(self, "final_url", _public_provenance_url(self.final_url))
        object.__setattr__(
            self,
            "source_urls",
            list(
                dict.fromkeys(
                    _public_provenance_url(value) for value in self.source_urls
                )
            ),
        )
        if self.tier is not None and self.tier < 0:
            raise ValueError("Provenance.tier must not be negative")
        if not self.engine.strip():
            raise ValueError("Provenance.engine must not be empty")
        if not self.recipe.strip():
            raise ValueError("Provenance.recipe must not be empty")
        if not self.canonical_url or not self.final_url:
            raise ValueError("Provenance URLs must not be empty")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ValueError("Provenance.http_status must be a valid HTTP status")
        if not self.source_urls or any(not value.strip() for value in self.source_urls):
            raise ValueError("Provenance.source_urls must contain non-empty URLs")
        try:
            parsed = datetime.fromisoformat(self.fetched_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Provenance.fetched_at must be an ISO-8601 string") from exc
        if parsed.tzinfo is None:
            raise ValueError("Provenance.fetched_at must include a timezone")


@dataclass(frozen=True, slots=True)
class ReadResult:
    """The complete response contract returned by every reader tier and recipe."""

    url: str
    content: str
    outline: Outline
    completeness: Completeness
    provenance: Provenance
    structured_data: JsonValue
    coverage: Coverage
    cost_to_complete: CostToComplete | None
    truncated: bool
    omitted: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _public_provenance_url(self.url))
        if not self.url.strip():
            raise ValueError("ReadResult.url must not be empty")
        if self.coverage not in get_args(Coverage):
            raise ValueError(f"Invalid coverage: {self.coverage!r}")
        if self.truncated and not self.omitted:
            raise ValueError("A truncated ReadResult must name at least one omitted section")
        if not self.truncated and self.omitted:
            raise ValueError("An untruncated ReadResult cannot contain omitted sections")
        if self.coverage == "core":
            if not self.truncated or self.cost_to_complete is None:
                raise ValueError(
                    "Core coverage must be explicitly truncated and name its cost to complete"
                )
        elif self.truncated:
            raise ValueError("Full coverage cannot carry deliberate omissions")
        elif (
            self.cost_to_complete is not None
            and self.completeness.status != "incomplete"
        ):
            raise ValueError(
                "Full coverage can carry completion cost only for a known incomplete capture"
            )


def _public_provenance_url(value: str) -> str:
    """Remove query and fragment data that may contain session-bearing secrets."""

    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


class OmniReadError(Exception):
    """Base class for all expected OmniRead failures."""


class RetrievalError(OmniReadError):
    """The source could not be fetched or returned an unusable HTTP response."""


class BlockedError(RetrievalError):
    """The response was a block or challenge page, including deceptive HTTP 200s."""


class RenderError(OmniReadError):
    """A browser tier could not render the requested page."""


class PolicyError(OmniReadError):
    """A configured access policy rejected the request."""


class ExtractionError(OmniReadError):
    """A fetched representation yielded no usable content."""


class VerificationError(OmniReadError):
    """The verifier itself failed to evaluate the available evidence."""


class TruncationError(OmniReadError):
    """Content could not be truncated without violating a section boundary."""


class BudgetError(OmniReadError):
    """A requested token budget is invalid or cannot be applied."""


class DependencyError(OmniReadError):
    """An optional adopted engine required for this read is not installed."""
