"""Failure-classified generic escalation ladder for static and rendered pages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from ._html import has_pagination
from .extract import ExtractedPage, extract_page
from .fetch import FetchedPage, _validate_http_url, detect_block_page, fetch_static
from .renderer import DefuddleRenderer, render_page
from .tokens import build_outline, cost_for_omitted_sections, truncate_to_budget
from .types import (
    Completeness,
    Evidence,
    ExtractionError,
    OmniReadError,
    PolicyError,
    Provenance,
    ReadResult,
    RetrievalError,
    VerificationError,
)
from .verify import verify_completeness

type Clock = Callable[[], datetime | str]
type Renderer = Callable[[str], FetchedPage]


class AttemptFailure(StrEnum):
    """Why one tier did not produce a verified-complete representation."""

    RETRIEVAL = "retrieval"
    BLOCK = "block"
    AUTH = "auth"
    EXTRACTION = "extraction"
    VERIFICATION = "verification"
    PAGINATION = "pagination"


@dataclass(frozen=True, slots=True)
class TierAttempt:
    """The internal seam consumed by the next tier in a future dispatch."""

    tier: int
    engine: str
    fetched: FetchedPage | None
    extracted: ExtractedPage | None
    completeness: Completeness | None
    failure: AttemptFailure | None
    error: OmniReadError | None


def read(
    url: str,
    *,
    budget: int | None = None,
    full: bool = False,
    clock: Clock | None = None,
    render: Renderer | None = None,
    auth_profile: Path | None = None,
    auth_render: Renderer | None = None,
    browser_auth: bool = False,
    auth_required: bool = False,
    login_command: str | None = None,
) -> ReadResult:
    """Read a URL through the generic ladder and return the typed contract.

    Tier 0 is a companion probe inside extraction, never a generic shortcut. Tier 2
    renders a likely JavaScript shell or one bot challenge. Pagination does not
    trigger rendering. The caller injects a clock for deterministic provenance.
    """

    if clock is None:
        raise ValueError("read() requires a caller-supplied clock")

    if auth_required:
        if auth_profile is None and not (browser_auth and auth_render is not None):
            raise PolicyError(
                "This site requires the user's own login profile. "
                f"Run {login_command or f'omniread login {url}'} first."
            )
        authenticated = auth_render or DefuddleRenderer(
            user_data_dir=auth_profile
        ).render
        attempt = _attempt_tier4(url, authenticated)
    else:
        attempt = _attempt_tier1(url)
        if _should_render(attempt):
            renderer = render or render_page
            rendered = _attempt_tier2(url, renderer)
            if _render_improved(attempt, rendered):
                attempt = rendered
        if (
            (auth_profile is not None or browser_auth)
            and _should_try_authenticated(attempt)
        ):
            authenticated = auth_render or DefuddleRenderer(
                user_data_dir=auth_profile
            ).render
            attempt = _choose_authenticated(
                attempt, _attempt_tier4(url, authenticated)
            )

    return _result_from_attempt(
        url, attempt, budget=budget, full=full, clock=clock, login_command=login_command,
    )


def _result_from_attempt(
    url: str, attempt: TierAttempt, *, budget: int | None, full: bool,
    clock: Clock, login_command: str | None = None,
) -> ReadResult:
    """Apply the shared verdict, auth scrub, outline and budget contract."""

    if attempt.error is not None:
        raise attempt.error
    if attempt.fetched is None or attempt.extracted is None:
        raise RetrievalError("Tier 1 ended without fetch/extraction state")

    if attempt.completeness is None:
        raise VerificationError("Tier 1 ended without a completeness verdict")
    completeness = attempt.completeness
    if attempt.failure is AttemptFailure.AUTH:
        command = login_command or f"omniread login {url}"
        state = "stored login did not clear" if attempt.tier == 4 else "no stored profile can clear"
        completeness = Completeness(
            status="unknown",
            evidence=completeness.evidence,
            reason=(
                f"Authentication is required and {state} the login wall. "
                f"Run {command}, then retry."
            ),
        )
    auth_blocked = attempt.failure is AttemptFailure.AUTH
    output_markdown = "" if auth_blocked else attempt.extracted.markdown
    # On an auth wall, scrub the structured-data side channel too: publishers commonly embed
    # the full gated article in JSON-LD (articleBody), which would otherwise leak past the
    # empty-content scrub even though the verdict is honestly unknown.
    output_structured = {} if auth_blocked else attempt.extracted.structured_data
    outline = build_outline(
        output_markdown,
        has_pagination=has_pagination(
            attempt.fetched.raw_html,
            current_url=attempt.fetched.final_url,
        ),
    )
    bounded = truncate_to_budget(output_markdown, None if full else budget)
    coverage = "core" if bounded.truncated else "full"
    render_latency = attempt.fetched.headers.get("x-omniread-render-ms")
    estimated_latency_ms = (
        int(render_latency) if render_latency and render_latency.isdigit() else None
    )
    cost = (
        cost_for_omitted_sections(
            outline,
            bounded.omitted,
            required_tier=attempt.tier if attempt.tier >= 2 else None,
            estimated_latency_ms=estimated_latency_ms if attempt.tier >= 2 else None,
        )
        if bounded.truncated
        else None
    )
    fetched_at = clock_iso(clock)
    provenance = Provenance(
        tier=attempt.tier,
        engine=attempt.engine,
        recipe="generic",
        canonical_url=attempt.extracted.canonical_url,
        fetched_at=fetched_at,
        http_status=None if (
            attempt.fetched.headers.get("x-omniread-policy")
            or (attempt.tier == 4 and attempt.fetched.headers.get("x-omniread-http-status-unknown"))
        ) else attempt.fetched.http_status,
        final_url=attempt.fetched.final_url,
        source_urls=list(
            dict.fromkeys(
                attempt.fetched.source_urls or (url, attempt.fetched.final_url)
            )
        ),
    )
    return ReadResult(
        url=url,
        content=bounded.content,
        outline=outline,
        completeness=completeness,
        provenance=provenance,
        structured_data=output_structured,
        coverage=coverage,
        cost_to_complete=cost,
        truncated=bounded.truncated,
        omitted=bounded.omitted,
    )


def read_html(
    html: str, url: str, *, budget: int | None = None, full: bool = False, clock: Clock,
) -> ReadResult:
    """Verify supplied HTML through the generic contract, without fetching or rendering.

    Completeness refers to this representation. Transport status and retrieval tier
    are unknown because the caller supplies only HTML and its base URL.
    """

    _validate_http_url(url)
    if not isinstance(html, str):
        raise TypeError("HTML must be a string")
    page = FetchedPage(
        raw_html=html, http_status=200, final_url=url,
        headers={"content-type": "text/html"},
        block_signal=detect_block_page(html, http_status=200, final_url=url),
        source_urls=(url,),
    )
    attempt = _attempt(
        url, tier=1, engine="supplied-html+trafilatura", fetcher=lambda _: page, allow_empty=True,
    )
    result = _result_from_attempt(url, attempt, budget=budget, full=full, clock=clock)
    return replace(
        result,
        provenance=replace(result.provenance, tier=None, http_status=None),
        completeness=replace(result.completeness, evidence=[
            *result.completeness.evidence,
            Evidence("supplied_html", None,
                     "Verified the supplied HTML only; HTTP status and upstream capture completeness were not observed"),
        ]),
    )


def _attempt_tier1(url: str) -> TierAttempt:
    return _attempt(
        url,
        tier=1,
        engine="curl_cffi+trafilatura",
        fetcher=fetch_static,
    )


def _attempt_tier2(url: str, render: Renderer) -> TierAttempt:
    return _attempt(
        url,
        tier=2,
        engine="playwright-chromium+trafilatura",
        fetcher=render,
    )


def _attempt_tier4(url: str, render: Renderer) -> TierAttempt:
    return _attempt(
        url,
        tier=4,
        engine="playwright-chromium-persistent-profile+trafilatura",
        fetcher=render,
    )


def _attempt(
    url: str,
    *,
    tier: int,
    engine: str,
    fetcher: Callable[[str], FetchedPage],
    allow_empty: bool = False,
) -> TierAttempt:
    renderer = getattr(fetcher, "__self__", fetcher)
    engine = getattr(renderer, "engine", engine)
    try:
        fetched = fetcher(url)
    except (OmniReadError, PolicyError) as exc:
        return TierAttempt(
            tier=tier,
            engine=engine,
            fetched=None,
            extracted=None,
            completeness=None,
            failure=AttemptFailure.RETRIEVAL,
            error=exc,
        )

    if tier >= 2:
        if fetched.headers.get("x-omniread-render-tier") == "4":
            tier = 4
            engine = "playwright-chromium-persistent-profile+trafilatura"
        engine = fetched.headers.get("x-omniread-render-engine", engine)

    if not 200 <= fetched.http_status < 300 and not fetched.block_signal.detected:
        error = RetrievalError(
            f"Tier-{tier} HTTP status {fetched.http_status} for {fetched.final_url}"
        )
        return TierAttempt(
            tier=tier,
            engine=engine,
            fetched=fetched,
            extracted=None,
            completeness=None,
            failure=AttemptFailure.RETRIEVAL,
            error=error,
        )

    try:
        extracted = extract_page(fetched.raw_html, url=fetched.final_url)
    except Exception as exc:
        error = ExtractionError(f"Tier-{tier} Trafilatura extraction failed: {exc}")
        return TierAttempt(
            tier=tier,
            engine=engine,
            fetched=fetched,
            extracted=None,
            completeness=None,
            failure=AttemptFailure.EXTRACTION,
            error=error,
        )

    if not extracted.markdown and not fetched.block_signal.detected and not allow_empty:
        error = ExtractionError(
            f"Tier-{tier} fetch succeeded but Trafilatura returned no content"
        )
        return TierAttempt(
            tier=tier,
            engine=engine,
            fetched=fetched,
            extracted=extracted,
            completeness=None,
            failure=AttemptFailure.EXTRACTION,
            error=error,
        )

    try:
        verdict = verify_completeness(fetched, extracted)
        if fetched.headers.get("x-omniread-capture-truncated") == "true" and not fetched.block_signal.detected:
            verdict = Completeness(
                "incomplete",
                [*verdict.evidence, Evidence("capture_truncation", False,
                    "Agent Browser reported truncating the HTML capture at its size limit")],
                "The browser supplied a truncated HTML capture",
            )
    except Exception as exc:
        error = VerificationError(f"Completeness verification failed: {exc}")
        return TierAttempt(
            tier=tier,
            engine=engine,
            fetched=fetched,
            extracted=extracted,
            completeness=None,
            failure=AttemptFailure.VERIFICATION,
            error=error,
        )
    pagination_gap = any(
        signal.name == "pagination_coverage" and signal.passed is False
        for signal in verdict.evidence
    )
    if fetched.block_signal.auth_required:
        failure = AttemptFailure.AUTH
    elif fetched.block_signal.detected:
        failure = AttemptFailure.BLOCK
    elif pagination_gap:
        failure = AttemptFailure.PAGINATION
    else:
        failure = None if verdict.status == "complete" else AttemptFailure.VERIFICATION
    return TierAttempt(
        tier=tier,
        engine=engine,
        fetched=fetched,
        extracted=extracted,
        completeness=verdict,
        failure=failure,
        error=None,
    )


def _should_render(attempt: TierAttempt) -> bool:
    """Escalate a plausible client-rendered shell, or a bot challenge, never a known wrong class."""

    if attempt.failure in {AttemptFailure.AUTH, AttemptFailure.PAGINATION}:
        return False
    # A bot/challenge interstitial is the one failure class a real browser exists
    # to clear: Tier 1 sends no browser at all, so declining to render it reported
    # readable pages as blocked. Render once; _render_improved keeps Tier 1's
    # evidence when the browser is challenged too.
    if attempt.failure is AttemptFailure.BLOCK:
        return True
    if attempt.fetched is None or not _looks_like_js_shell(attempt.fetched.raw_html):
        return False
    if attempt.failure is AttemptFailure.EXTRACTION:
        return True
    return (
        attempt.failure is AttemptFailure.VERIFICATION
        and attempt.completeness is not None
        and attempt.completeness.status == "unknown"
    )


def _render_improved(attempt: TierAttempt, rendered: TierAttempt) -> bool:
    """Adopt a Tier-2 attempt only when it carries state Tier 1 did not."""

    if (
        rendered.error is not None
        or rendered.fetched is None
        or rendered.extracted is None
    ):
        return False
    # A browser that is challenged in turn proves nothing the static tier had not
    # already recorded, and swapping it in would hide Tier 1's block evidence.
    return not (
        attempt.failure is AttemptFailure.BLOCK
        and rendered.failure is AttemptFailure.BLOCK
        and not rendered.fetched.headers.get("x-omniread-policy")
    )


def _looks_like_js_shell(raw_html: str) -> bool:
    lowered = raw_html.lower()
    script_signal = "<script" in lowered
    mount_signal = any(
        marker in lowered
        for marker in ('id="app"', "id='app'", 'id="root"', "id='root'", "__next")
    )
    return script_signal and mount_signal


# Failures a stored per-site login profile might legitimately clear. A login cannot fix
# pagination ("page 1 of N" is not a wall) or a hard retrieval/extraction error, so those
# never trigger an authenticated retry.
_AUTH_RETRY_FAILURES = frozenset(
    {AttemptFailure.AUTH, AttemptFailure.BLOCK, AttemptFailure.VERIFICATION}
)


def _should_try_authenticated(attempt: TierAttempt) -> bool:
    """Escalate to a stored-profile render whenever we did not reach complete content.

    Language-agnostic on purpose: a non-English paywall is classified as a plain
    ``incomplete`` content gap (``AttemptFailure.VERIFICATION``), never ``AUTH``, so keying
    the retry on English marker text would miss it. Instead, the user having run
    ``omniread login`` for this domain (an ``auth_profile`` is present) is the bounded signal
    that one authenticated attempt is worth making.
    """

    if attempt.tier == 4 or attempt.error is not None:
        return False
    if attempt.completeness is not None and attempt.completeness.status == "complete":
        return False
    return attempt.failure in _AUTH_RETRY_FAILURES


def _choose_authenticated(
    anonymous: TierAttempt, authenticated: TierAttempt
) -> TierAttempt:
    """Prefer the authenticated view the user logged in for, but only when it is genuinely better.

    Each attempt keeps its own INDEPENDENT completeness verdict, so preferring the
    authenticated one can never turn a partial into a false ``complete``. The real risk is the
    opposite: a tripped or expired session returns a bot challenge or an empty stub, and
    blindly preferring it would replace usable anonymous content with something worse. So the
    authenticated attempt wins only when it verified ``complete``, or when it carries real,
    non-blocked content. NB: length is NOT the test — a legitimate logged-in view is often
    SHORTER than the anonymous teaser because the paywall boilerplate is gone.
    """

    if authenticated.error is not None:
        return anonymous
    if (
        authenticated.completeness is not None
        and authenticated.completeness.status == "complete"
    ):
        return authenticated
    fetched = authenticated.fetched
    if fetched is not None and fetched.headers.get("x-omniread-policy") and anonymous.completeness is not None:
        detail = "; ".join(fetched.block_signal.reasons)
        return replace(anonymous, completeness=replace(
            anonymous.completeness,
            evidence=[*anonymous.completeness.evidence, Evidence("agent_browser_policy", False, detail)],
            reason=f"{anonymous.completeness.reason}. {detail}",
        ))
    if fetched is None or fetched.block_signal.detected:
        return anonymous
    body = authenticated.extracted.markdown if authenticated.extracted else ""
    if not body.strip():
        return anonymous
    return authenticated


def clock_iso(clock: Clock) -> str:
    """Read an injected clock into the contract's ISO-8601 form."""

    value = clock()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("The caller-supplied clock returned a naive datetime")
        return value.isoformat()
    if isinstance(value, str):
        return value
    raise TypeError("The caller-supplied clock must return datetime or ISO-8601 str")
