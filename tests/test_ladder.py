from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from omniread import ladder
from omniread.fetch import FetchedPage, detect_block_page
from omniread.tokens import build_outline
from omniread.types import ExtractionError, RenderError, RetrievalError

FIXTURES = Path(__file__).parent / "fixtures"
FIXED_TIME = datetime(2026, 7, 12, 13, 30, tzinfo=timezone.utc)


def _page(file_name: str, *, status: int = 200) -> FetchedPage:
    raw_html = (FIXTURES / file_name).read_text(encoding="utf-8")
    return FetchedPage(
        raw_html=raw_html,
        http_status=status,
        final_url="https://example.test/final",
        headers={"content-type": "text/html"},
        block_signal=detect_block_page(raw_html, http_status=status),
    )


def test_tier1_read_returns_full_typed_result(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("normal_article.html"))

    result = ladder.read("https://example.test/start", clock=lambda: FIXED_TIME)

    assert result.completeness.status == "complete"
    assert result.provenance.tier == 1
    assert result.provenance.engine == "curl_cffi+trafilatura"
    assert result.provenance.final_url == "https://example.test/final"
    assert result.provenance.canonical_url == "https://example.test/articles/reliable-reader"
    assert result.provenance.fetched_at == FIXED_TIME.isoformat()
    assert result.structured_data is not None
    assert result.coverage == "full"
    assert result.cost_to_complete is None
    assert result.truncated is False
    assert result.omitted == []


def test_client_redirect_hops_are_preserved_in_result_provenance(monkeypatch) -> None:
    page = _page("normal_article.html")
    redirected = FetchedPage(
        raw_html=page.raw_html,
        http_status=page.http_status,
        final_url="https://example.test/article/?session=secret#account",
        headers=page.headers,
        block_signal=page.block_signal,
        source_urls=(
            "https://example.test/article.html?token=secret",
            "https://example.test/article/?session=secret#account",
        ),
    )
    monkeypatch.setattr(ladder, "fetch_static", lambda url: redirected)

    result = ladder.read("https://example.test/article.html", clock=lambda: FIXED_TIME)

    assert result.provenance.source_urls == [
        "https://example.test/article.html",
        "https://example.test/article/",
    ]
    assert result.provenance.final_url == "https://example.test/article/"


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("truncated_article.html", "incomplete"),
        ("block_page.html", "unknown"),
    ],
)
def test_unresolved_tier1_verdicts_surface_honestly(monkeypatch, fixture: str, expected: str) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page(fixture))

    result = ladder.read("https://example.test/start", clock=lambda: FIXED_TIME)

    assert result.completeness.status == expected


def test_ladder_classifies_block_before_future_escalation(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("block_page.html"))

    attempt = ladder._attempt_tier1("https://example.test/start")

    assert attempt.failure is ladder.AttemptFailure.BLOCK
    assert attempt.error is None


def test_ladder_classifies_verifier_gap_before_future_escalation(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("truncated_article.html"))

    attempt = ladder._attempt_tier1("https://example.test/start")

    assert attempt.failure is ladder.AttemptFailure.VERIFICATION
    assert attempt.error is None


def test_budget_is_applied_after_full_document_verification(monkeypatch) -> None:
    page = _page("tail_heavy.html")
    monkeypatch.setattr(ladder, "fetch_static", lambda url: page)
    full = ladder.read("https://example.test/start", clock=lambda: FIXED_TIME)
    first_section_budget = full.outline.sections[0].token_count

    bounded = ladder.read(
        "https://example.test/start",
        budget=first_section_budget,
        clock=lambda: FIXED_TIME,
    )

    assert bounded.completeness.status == "complete"
    assert bounded.coverage == "core"
    assert bounded.cost_to_complete is not None
    assert bounded.cost_to_complete.remaining_items == len(full.outline.sections) - 1
    assert bounded.cost_to_complete.estimated_extra_tokens == sum(
        section.token_count for section in full.outline.sections[1:]
    )
    assert bounded.cost_to_complete.required_tier is None
    assert bounded.cost_to_complete.estimated_latency_ms is None
    assert bounded.truncated is True
    assert bounded.omitted == [section.anchor for section in full.outline.sections[1:]]
    assert build_outline(bounded.content).sections[0].anchor == full.outline.sections[0].anchor


def test_browser_core_cost_names_the_required_tier_and_measured_latency(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("js_spa_shell.html"))

    def render(url: str) -> FetchedPage:
        page = _page("rendered_spa.html")
        return FetchedPage(
            raw_html=page.raw_html,
            http_status=page.http_status,
            final_url=page.final_url,
            headers={**page.headers, "x-omniread-render-ms": "1350"},
            block_signal=page.block_signal,
        )

    full = ladder.read(
        "https://example.test/app",
        full=True,
        clock=lambda: FIXED_TIME,
        render=render,
    )
    core = ladder.read(
        "https://example.test/app",
        budget=full.outline.sections[0].token_count,
        clock=lambda: FIXED_TIME,
        render=render,
    )

    assert core.cost_to_complete is not None
    assert core.cost_to_complete.required_tier == 2
    assert core.cost_to_complete.estimated_latency_ms == 1350


def test_full_flag_ignores_a_previous_core_budget(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("normal_article.html"))

    result = ladder.read(
        "https://example.test/start",
        budget=1,
        full=True,
        clock=lambda: FIXED_TIME,
    )

    assert result.coverage == "full"
    assert result.truncated is False
    assert result.cost_to_complete is None


def test_js_shell_escalates_to_defuddle_bundled_chromium(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("js_spa_shell.html"))
    rendered_urls: list[str] = []

    def render(url: str) -> FetchedPage:
        rendered_urls.append(url)
        return _page("rendered_spa.html")

    result = ladder.read(
        "https://example.test/app",
        clock=lambda: FIXED_TIME,
        render=render,
    )

    assert rendered_urls == ["https://example.test/app"]
    assert result.completeness.status == "complete"
    assert result.provenance.tier == 2
    assert result.provenance.engine == "playwright-chromium+trafilatura"


def test_persistent_renderer_header_cannot_be_mislabeled_as_tier2(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("js_spa_shell.html"))

    def render(url: str) -> FetchedPage:
        page = _page("rendered_spa.html")
        return FetchedPage(
            raw_html=page.raw_html,
            http_status=page.http_status,
            final_url=page.final_url,
            headers={**page.headers, "x-omniread-render-tier": "4"},
            block_signal=page.block_signal,
        )

    result = ladder.read(
        "https://example.test/app",
        clock=lambda: FIXED_TIME,
        render=render,
    )

    assert result.provenance.tier == 4
    assert "persistent-profile" in result.provenance.engine


def test_pagination_gap_does_not_waste_a_browser_render(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("paginated_thread.html"))
    rendered = False

    def render(url: str) -> FetchedPage:
        nonlocal rendered
        rendered = True
        return _page("paginated_thread.html")

    result = ladder.read(
        "https://example.test/thread",
        clock=lambda: FIXED_TIME,
        render=render,
    )

    assert rendered is False
    assert result.completeness.status == "incomplete"
    assert result.coverage == "full"


def test_read_requires_caller_supplied_clock_before_fetch(monkeypatch) -> None:
    called = False

    def fake_fetch(url: str):
        nonlocal called
        called = True
        return _page("normal_article.html")

    monkeypatch.setattr(ladder, "fetch_static", fake_fetch)

    with pytest.raises(ValueError, match="caller-supplied clock"):
        ladder.read("https://example.test/start")
    assert called is False


def test_non_block_http_failure_raises_typed_retrieval_error(monkeypatch) -> None:
    monkeypatch.setattr(
        ladder,
        "fetch_static",
        lambda url: FetchedPage(
            raw_html="<html><title>Unavailable</title><body>Try later</body></html>",
            http_status=503,
            final_url=url,
            headers={},
            block_signal=detect_block_page("<html><body>Try later</body></html>", http_status=503),
        ),
    )

    with pytest.raises(RetrievalError, match="HTTP status 503"):
        ladder.read("https://example.test/start", clock=lambda: FIXED_TIME)


def test_empty_successful_page_raises_typed_extraction_error(monkeypatch) -> None:
    raw_html = "<html><head><title>Empty</title></head><body></body></html>"
    monkeypatch.setattr(
        ladder,
        "fetch_static",
        lambda url: FetchedPage(
            raw_html=raw_html,
            http_status=200,
            final_url=url,
            headers={},
            block_signal=detect_block_page(raw_html, http_status=200),
        ),
    )

    with pytest.raises(ExtractionError, match="returned no content"):
        ladder.read("https://example.test/start", clock=lambda: FIXED_TIME)


def test_content_gap_with_profile_escalates_to_authenticated_tier4(monkeypatch, tmp_path) -> None:
    # A non-English paywall reads as a plain content gap (not AUTH); a stored profile must
    # still trigger an authenticated retry that reveals the full body.
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("truncated_article.html"))
    calls: list[str] = []

    def auth_render(url: str) -> FetchedPage:
        calls.append(url)
        return _page("normal_article.html")

    result = ladder.read(
        "https://example.test/locked",
        clock=lambda: FIXED_TIME,
        auth_profile=tmp_path / "profile",
        auth_render=auth_render,
    )

    assert calls == ["https://example.test/locked"]  # tier 4 attempted exactly once
    assert result.provenance.tier == 4
    assert "persistent-profile" in result.provenance.engine
    assert result.completeness.status == "complete"


def test_content_gap_without_profile_does_not_escalate(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("truncated_article.html"))
    called = False

    def auth_render(url: str) -> FetchedPage:
        nonlocal called
        called = True
        return _page("normal_article.html")

    result = ladder.read(
        "https://example.test/locked",
        clock=lambda: FIXED_TIME,
        auth_render=auth_render,  # provided, but no auth_profile => must never be used
    )

    assert called is False
    assert result.provenance.tier == 1
    assert result.completeness.status == "incomplete"


def test_authenticated_retry_that_is_still_partial_never_reports_complete(monkeypatch, tmp_path) -> None:
    # e.g. an audio-gated podcast: login removes the paywall chrome but there is no text to
    # reveal. The authenticated view is preferred, yet the verdict stays honest.
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("truncated_article.html"))

    def auth_render(url: str) -> FetchedPage:
        return _page("truncated_article.html")

    result = ladder.read(
        "https://example.test/locked",
        clock=lambda: FIXED_TIME,
        auth_profile=tmp_path / "profile",
        auth_render=auth_render,
    )

    assert result.provenance.tier == 4
    assert result.completeness.status == "incomplete"


def test_pagination_gap_with_profile_does_not_trigger_authenticated_retry(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("paginated_thread.html"))
    called = False

    def auth_render(url: str) -> FetchedPage:
        nonlocal called
        called = True
        return _page("normal_article.html")

    result = ladder.read(
        "https://example.test/thread",
        clock=lambda: FIXED_TIME,
        auth_profile=tmp_path / "profile",
        auth_render=auth_render,
    )

    assert called is False  # a login cannot fix "page 1 of N"
    assert result.completeness.status == "incomplete"


def test_authenticated_block_does_not_replace_anonymous_content(monkeypatch, tmp_path) -> None:
    # A stored session that trips a bot challenge must NOT overwrite usable anonymous content
    # with the challenge stub (a block is not an error, so the error-only guard missed this).
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("truncated_article.html"))

    def auth_render(url: str) -> FetchedPage:
        return _page("block_page.html")

    result = ladder.read(
        "https://example.test/locked",
        clock=lambda: FIXED_TIME,
        auth_profile=tmp_path / "profile",
        auth_render=auth_render,
    )

    assert result.provenance.tier == 1  # anonymous partial content kept
    assert result.completeness.status == "incomplete"
    assert result.content.strip() != ""


def test_authenticated_render_error_falls_back_to_anonymous(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("truncated_article.html"))

    def auth_render(url: str) -> FetchedPage:
        raise RenderError("bundled Chromium failed")

    result = ladder.read(
        "https://example.test/locked",
        clock=lambda: FIXED_TIME,
        auth_profile=tmp_path / "profile",
        auth_render=auth_render,
    )

    assert result.provenance.tier == 1
    assert result.completeness.status == "incomplete"
    assert result.content.strip() != ""


def test_still_walled_after_tier4_stays_unknown_and_empty(monkeypatch, tmp_path) -> None:
    # An expired/ineffective login: the authenticated render still hits the wall. The verdict
    # must stay unknown with empty content and a login instruction — never leak a partial.
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("authwall_page.html"))

    def auth_render(url: str) -> FetchedPage:
        return _page("authwall_page.html")

    result = ladder.read(
        "https://example.test/locked",
        clock=lambda: FIXED_TIME,
        auth_profile=tmp_path / "profile",
        auth_render=auth_render,
    )

    assert result.completeness.status == "unknown"
    assert result.content == ""
    assert "omniread login" in result.completeness.reason


def test_auth_wall_scrubs_structured_data_side_channel(monkeypatch) -> None:
    # The empty-content scrub must extend to structured_data, or a JSON-LD articleBody would
    # leak the full gated article even though content is "" and the verdict is unknown.
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("authwall_page.html"))

    result = ladder.read("https://example.test/locked", clock=lambda: FIXED_TIME)

    assert result.completeness.status == "unknown"
    assert result.content == ""
    assert result.structured_data == {}
