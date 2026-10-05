from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omniread import fetch

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_normal_article_is_not_mistaken_for_a_block_page() -> None:
    signal = fetch.detect_block_page(_fixture("normal_article.html"), http_status=200)
    assert signal.detected is False
    assert signal.reasons == ()


def test_http_200_challenge_is_signaled_without_raising() -> None:
    signal = fetch.detect_block_page(_fixture("block_page.html"), http_status=200)
    assert signal.detected is True
    assert signal.reasons[0] == "HTTP 200 response contains block-page evidence"
    assert any("block title" in reason for reason in signal.reasons)


def test_authwall_is_distinct_from_generic_block_and_hidden_scaffold_is_ignored() -> None:
    authwall = fetch.detect_block_page(_fixture("authwall_page.html"), http_status=200)
    ordinary = fetch.detect_block_page(
        """
        <html><body><main><h1>Short public note</h1><p>This is the entire public note.</p>
        <div class="paywall" hidden>Subscribe to continue reading</div></main></body></html>
        """,
        http_status=200,
    )
    quoted = fetch.detect_block_page(
        """
        <html><body><main><h1>Paywall design notes</h1>
        <p>The phrase subscribe to continue reading is a common paywall pattern.</p>
        </main></body></html>
        """,
        http_status=200,
    )
    short_public_page = fetch.detect_block_page(
        """
        <html><body><h1>Public glossary entry</h1>
        <p>Subscribe to continue reading is the phrase this entry defines.</p>
        <p>This complete public page intentionally has no article or main element.</p>
        </body></html>
        """,
        http_status=200,
    )
    login_form = fetch.detect_block_page(
        """
        <html><body><p>Sign in to continue reading.</p>
        <form action="/login"><input type="password" name="password"></form>
        </body></html>
        """,
        http_status=200,
    )

    assert authwall.detected is True
    assert authwall.auth_required is True
    assert ordinary.detected is False
    assert ordinary.auth_required is False
    assert quoted.detected is False
    assert quoted.auth_required is False
    assert short_public_page.detected is False
    assert short_public_page.auth_required is False
    assert login_form.detected is True
    assert login_form.auth_required is True


def test_truncated_article_is_a_gap_not_misclassified_as_a_block() -> None:
    signal = fetch.detect_block_page(_fixture("truncated_article.html"), http_status=200)
    assert signal.detected is False


def test_fetch_uses_browser_tls_impersonation(monkeypatch) -> None:
    calls: dict[str, object] = {}

    @dataclass
    class Response:
        text: str = "<html><body><main><p>Body</p></main></body></html>"
        status_code: int = 200
        url: str = "https://example.test/final"
        headers: dict[str, str] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            self.headers = {"content-type": "text/html"}

    def fake_get(url: str, **kwargs):
        calls.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    result = fetch.fetch_static("https://example.test/start")

    assert calls["impersonate"] == "chrome"
    assert calls["allow_redirects"] is True
    assert result.final_url == "https://example.test/final"
    assert result.http_status == 200


def test_fetch_follows_one_thin_client_redirect_to_fixtured_content(monkeypatch) -> None:
    calls: list[str] = []
    responses = {
        "https://example.test/releases/1.0.html": (
            _fixture("client_redirect_stub.html"),
            "https://example.test/releases/1.0.html",
        ),
        "https://example.test/articles/reliable-reader": (
            _fixture("normal_article.html"),
            "https://example.test/articles/reliable-reader",
        ),
    }

    @dataclass
    class Response:
        text: str
        url: str
        status_code: int = 200
        headers: dict[str, str] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            self.headers = {"content-type": "text/html"}

    def fake_get(url: str, **kwargs):
        calls.append(url)
        body, final_url = responses[url]
        return Response(body, final_url)

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    result = fetch.fetch_static("https://example.test/releases/1.0.html")

    assert calls == [
        "https://example.test/releases/1.0.html",
        "https://example.test/articles/reliable-reader",
    ]
    assert result.raw_html == _fixture("normal_article.html")
    assert result.final_url == "https://example.test/articles/reliable-reader"
    assert result.source_urls == (
        "https://example.test/releases/1.0.html",
        "https://example.test/articles/reliable-reader",
    )


def test_client_redirect_signal_never_overrides_a_real_page(monkeypatch) -> None:
    raw_html = """
    <html><head><meta http-equiv="refresh" content="0; url=/replacement"></head>
    <body><main><h1>Real status page</h1><p>This main content is intentionally short but real.
    The redirect metadata is stale, so the static reader must preserve this representation
    instead of replacing it merely because a redirect-shaped tag exists.</p></main></body></html>
    """
    calls: list[str] = []

    @dataclass
    class Response:
        text: str = raw_html
        url: str = "https://example.test/status"
        status_code: int = 200
        headers: dict[str, str] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            self.headers = {"content-type": "text/html"}

    def fake_get(url: str, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    result = fetch.fetch_static("https://example.test/status")

    assert calls == ["https://example.test/status"]
    assert result.raw_html == raw_html
