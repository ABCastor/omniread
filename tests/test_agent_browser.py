from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import socketserver
import tempfile
import threading

import pytest

from omniread import agent_browser, cli, ladder
from omniread.auth import ProfileStore
from omniread.fetch import FetchedPage, detect_block_page
from omniread.reader import Reader
from omniread.recipes import RecipeRegistry
from omniread.recipes.generic import GenericRecipe
from omniread.recipes.linkedin import LinkedInRecipe
from omniread.renderer import DefuddleRenderer
from omniread.types import RenderError

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://example.test/article"
CLOCK = lambda: "2026-09-12T12:00:00+00:00"


class FakeDaemon:
    def __init__(self, path: Path):
        self.path = path
        self.requests = []
        self.errors = {}
        self.overrides = {}
        self.html = (FIXTURES / "normal_article.html").read_text()
        self.final_url = URL
        self.status = 200
        daemon = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                request = json.loads(self.rfile.readline())
                daemon.requests.append(request)
                method = request["method"]
                tab = {
                    "id": 42,
                    "owned": True,
                    "url": daemon.final_url,
                    "title": "fake",
                    "active": True,
                    "windowId": 1,
                    "index": 0,
                }
                defaults = {
                    "status": {"browser": True},
                    "tabs": {"tabs": [dict(tab)]},
                    "open": {"tab": dict(tab)},
                    "html": {"tab": dict(tab), "url": daemon.final_url, "html": daemon.html, "status": daemon.status},
                    "close": {"closed": [42], "failed": []},
                }
                response = {"id": request["id"], "result": defaults[method]}
                if method in daemon.errors:
                    response = {"id": request["id"], "error": daemon.errors[method]}
                if method in daemon.overrides:
                    response = daemon.overrides[method](request)
                wire = response if isinstance(response, bytes) else (json.dumps(response) + "\n").encode()
                # Exercise framing across multiple writes, including JSON escapes.
                try:
                    self.wfile.write(wire[:17])
                    self.wfile.flush()
                    self.wfile.write(wire[17:])
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = socketserver.ThreadingUnixStreamServer(str(path), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()

    @property
    def methods(self):
        return [request["method"] for request in self.requests]


@pytest.fixture
def daemon(monkeypatch):
    # macOS Unix socket paths are limited to 104 bytes; pytest's tmp_path is longer.
    with ExitStack() as stack:
        directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="orab-", dir="/tmp"))
        fake = FakeDaemon(Path(directory) / "gaddi.sock")
        stack.callback(fake.close)
        monkeypatch.setenv("GADDI_SOCKET", str(fake.path))
        yield fake


def _page(name, url=URL):
    html = (FIXTURES / name).read_text()
    return FetchedPage(html, 200, url, {}, detect_block_page(html, http_status=200))


def _reader(tmp_path, registry=None, **kwargs):
    return Reader(registry=registry or RecipeRegistry((GenericRecipe(),)),
                  profiles=ProfileStore(tmp_path / "profiles"), **kwargs)


def test_success_closes_owned_tab_and_preserves_metadata(daemon):
    daemon.final_url = "https://example.test/final"
    daemon.html += "<!-- Unicode café 雪 -->"
    assert agent_browser.available()
    result = agent_browser.GaddiRenderer()(URL)
    assert daemon.methods == ["status", "open", "html", "close"]
    assert daemon.requests[1]["params"] == {"url": URL, "foreground": False, "caller": "omniread"}
    assert daemon.requests[2]["params"] == {"tab": 42}
    assert daemon.requests[3]["params"] == {"tabs": [42]}
    assert result.raw_html == daemon.html
    assert result.final_url == daemon.final_url
    assert result.http_status == 200
    assert result.source_urls == (URL, daemon.final_url)
    assert not result.block_signal.detected
    assert result.headers["x-omniread-render-engine"] == "gaddi+trafilatura"
    assert result.headers["x-omniread-render-tier"] == "4"
    assert int(result.headers["x-omniread-render-ms"]) >= 0


@pytest.mark.parametrize("code", ["held", "denied"])
def test_policy_navigation_fails_closed_without_retry(daemon, monkeypatch, tmp_path, code):
    daemon.errors["html"] = {"code": code, "message": "navigation blocked", "approval": {"id": "approval-42"}}
    monkeypatch.setattr(ladder, "fetch_static", lambda _: pytest.fail("An explicit login read must not fetch anonymously"))
    result = _reader(tmp_path).read(URL, clock=CLOCK, as_me=True)
    assert result.completeness.status == "unknown"
    assert result.content == ""
    assert result.provenance.http_status is None  # No fabricated website response.
    assert result.provenance.engine == "gaddi+trafilatura"
    detail = " ".join(e.detail for e in result.completeness.evidence)
    assert f"policy {code}" in detail and "approval.id=approval-42" in detail
    assert daemon.methods == ["status", "open", "html", "close"]


def test_hold_at_open_is_block_evidence(daemon):
    daemon.errors["open"] = {"code": "held", "message": "pending", "approval": {"id": "a1"}}
    page = agent_browser.GaddiRenderer()(URL)
    assert page.block_signal.detected
    assert "a1" in page.block_signal.reasons[0]
    assert daemon.methods == ["open"]


@pytest.mark.parametrize("wire", [b"not json\n", b'{"id":"wrong","result":{}}\n', b'{}'])
def test_malformed_response_is_typed_and_tab_closes(daemon, wire):
    daemon.overrides["html"] = lambda _: wire
    with pytest.raises(RenderError):
        agent_browser.GaddiRenderer()(URL)
    assert daemon.methods[-1] == "close"


def test_timeout_is_bounded_and_tab_closes(daemon):
    def slow_response(request):
        threading.Event().wait(0.15)
        return {"id": request["id"], "result": {}}
    daemon.overrides["html"] = slow_response
    with pytest.raises(RenderError, match="timed out"):
        agent_browser.GaddiRenderer(timeout_seconds=0.05)(URL)
    assert daemon.methods[-1] == "close"


def test_block_html_is_not_complete(daemon):
    daemon.html = (FIXTURES / "block_page.html").read_text()
    page = agent_browser.GaddiRenderer()(URL)
    assert page.http_status == 200
    assert page.block_signal.detected


@pytest.mark.parametrize("fixture", ["normal_article.html", "js_spa_shell.html", "block_page.html", "authwall_page.html"])
def test_public_read_never_touches_socket(daemon, monkeypatch, tmp_path, fixture):
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page(fixture))
    monkeypatch.setattr(DefuddleRenderer, "render", lambda self, _: _page("normal_article.html"))
    monkeypatch.setattr(agent_browser, "available", lambda: pytest.fail("Public read probed Chrome"))
    result = _reader(tmp_path).read(URL, clock=CLOCK)
    assert result.provenance.engine != "gaddi+trafilatura"
    assert daemon.requests == []

def test_absent_daemon_selects_defuddle(monkeypatch, tmp_path):
    assert not agent_browser.available()
    calls = []
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("js_spa_shell.html"))
    def render(self, url):
        calls.append(url)
        return _page("normal_article.html")
    monkeypatch.setattr(DefuddleRenderer, "render", render)
    result = _reader(tmp_path).read(URL, clock=CLOCK)
    assert calls == [URL]
    assert result.provenance.engine == "playwright-chromium+trafilatura"


def test_stale_socket_is_unavailable(daemon):
    daemon.overrides["status"] = lambda _: b"invalid\n"
    assert not agent_browser.available()


def test_old_renderer_override_cannot_send_public_reads_to_chrome(daemon, monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIREAD_RENDERER", "gaddi")
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("js_spa_shell.html"))
    monkeypatch.setattr(DefuddleRenderer, "render", lambda self, _: _page("normal_article.html"))
    assert _reader(tmp_path).read(URL, clock=CLOCK).provenance.tier == 2
    assert daemon.requests == []

def test_as_me_uses_chrome_without_anonymous_fetch(daemon, monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "fetch_static", lambda _: pytest.fail("Anonymous fetch"))
    result = _reader(tmp_path).read(URL, clock=CLOCK, as_me=True)
    assert result.completeness.status == "complete"
    assert result.provenance.tier == 4
    assert daemon.methods == ["status", "open", "html", "close"]

def test_login_records_preference_without_creating_profile(daemon, tmp_path, monkeypatch, capsys):
    root = tmp_path / "profiles"
    monkeypatch.setenv("OMNIREAD_PROFILE_DIR", str(root))
    assert cli.main(["login", "example.com"]) == 0
    assert capsys.readouterr().out.strip() == "Log in once in your own Chrome; OmniRead reads through it"
    assert (root / "example.com.chrome-login").is_file()
    assert not (root / "example.com").exists()
    assert daemon.methods == ["status"]


def test_login_preference_routes_later_auth_retry_through_chrome(daemon, monkeypatch, tmp_path):
    reader = _reader(tmp_path)
    reader.profiles.remember_chrome_login(URL)
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("authwall_page.html"))
    result = reader.read(URL, clock=CLOCK)
    assert result.provenance.tier == 4
    assert result.provenance.engine == "gaddi+trafilatura"
    assert result.completeness.status == "complete"
    assert not reader.profiles.profile_dir(URL).exists()
    assert daemon.methods == ["status", "open", "html", "close"]

def test_auth_required_recipe_uses_browser_capability(daemon, monkeypatch, tmp_path):
    url = "https://www.linkedin.com/in/example/"
    daemon.final_url = url
    monkeypatch.setenv("OMNIREAD_LINKEDIN_ENABLED", "1")
    def trap(_):
        pytest.fail("Private profile must not use anonymous fetch")
    monkeypatch.setattr(ladder, "fetch_static", trap)
    registry = RecipeRegistry((LinkedInRecipe(), GenericRecipe()))
    result = _reader(tmp_path, registry).read(url, clock=CLOCK)
    assert result.provenance.tier == 4
    assert result.completeness.status == "complete"
    assert not (tmp_path / "profiles").exists()


def test_injected_renderer_overrides_default(daemon, monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("js_spa_shell.html"))
    calls = []
    def render(url):
        calls.append(url)
        return _page("normal_article.html")
    assert _reader(tmp_path, render=render).read(URL, clock=CLOCK).completeness.status == "complete"
    assert calls == [URL]
    assert daemon.methods == []


@pytest.mark.parametrize("code", ["held", "denied"])
def test_auth_policy_evidence_survives_retained_partial(daemon, monkeypatch, tmp_path, code):
    daemon.errors["html"] = {"code": code, "message": "policy stopped navigation", "approval": {"id": "a2"}}
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("truncated_article.html"))
    reader = _reader(tmp_path)
    reader.profiles.profile_dir(URL).mkdir(parents=True)
    result = reader.read(URL, clock=CLOCK)
    assert result.completeness.status == "incomplete"
    assert result.content  # Keep the useful anonymous representation.
    assert result.provenance.tier == 1
    assert any(e.name == "agent_browser_policy" and code in e.detail for e in result.completeness.evidence)
    assert daemon.methods.count("html") == 1
    assert daemon.methods[-1] == "close"


def test_explicit_auth_factory_still_overrides_browser(daemon, monkeypatch, tmp_path):
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir(URL).mkdir(parents=True)
    calls = []
    def factory(path):
        calls.append(path)
        return lambda _: _page("normal_article.html")
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("authwall_page.html"))
    result = _reader(tmp_path, auth_renderer_factory=factory).read(URL, clock=CLOCK)
    assert result.completeness.status == "complete"
    assert calls == [profiles.profile_dir(URL)]
    assert daemon.methods == []


def test_socket_discovery_precedence(monkeypatch):
    monkeypatch.setenv("GADDI_SOCKET", "/tmp/explicit.sock")
    assert agent_browser.socket_path() == Path("/tmp/explicit.sock")
    monkeypatch.delenv("GADDI_SOCKET")
    monkeypatch.setattr(Path, "exists", lambda path: True)
    assert agent_browser.socket_path() == Path.home() / "Library/Application Support/Gaddi/gaddi.sock"

@pytest.mark.parametrize("as_me", [False, True])
def test_missing_socket_uses_stored_profile(monkeypatch, tmp_path, as_me):
    reader = _reader(tmp_path)
    profile = reader.profiles.profile_dir(URL)
    profile.mkdir(parents=True)
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("authwall_page.html"))
    calls = []
    def render(self, url):
        calls.append((self.user_data_dir, url))
        return _page("normal_article.html")
    monkeypatch.setattr(DefuddleRenderer, "render", render)
    result = reader.read(URL, clock=CLOCK, as_me=as_me)
    assert result.completeness.status == "complete"
    assert result.provenance.tier == 4
    assert calls == [(profile, URL)]

@pytest.mark.parametrize("failed_html", [False, True])
def test_failed_browser_attempt_is_not_repeated_as_auth(daemon, monkeypatch, tmp_path, failed_html):
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("block_page.html"))
    daemon.html = (FIXTURES / "block_page.html").read_text()
    if failed_html:
        daemon.overrides["html"] = lambda _: b"bad json\n"
    monkeypatch.setattr(DefuddleRenderer, "render", lambda self, _: _page("block_page.html"))
    reader = _reader(tmp_path)
    reader.profiles.profile_dir(URL).mkdir(parents=True)
    result = reader.read(URL, clock=CLOCK)
    assert result.completeness.status == "unknown"
    assert result.provenance.tier == 1  # Preserve the original block evidence.
    assert daemon.methods.count("open") == 1
    assert daemon.methods.count("html") == 1
    assert daemon.methods[-1] == "close"


def test_static_headers_cannot_spoof_browser_provenance(monkeypatch, tmp_path):
    from dataclasses import replace
    page = _page("normal_article.html")
    page = replace(page, headers={"x-omniread-render-tier": "4", "x-omniread-render-engine": "gaddi+trafilatura"})
    monkeypatch.setattr(ladder, "fetch_static", lambda _: page)
    result = _reader(tmp_path).read(URL, clock=CLOCK)
    assert result.provenance.tier == 1
    assert result.provenance.engine == "curl_cffi+trafilatura"


@pytest.mark.parametrize("tab", [None, "", "42", True, -1])
def test_invalid_tab_is_not_closed(daemon, tab):
    daemon.overrides["open"] = lambda request: {"id": request["id"], "result": {"tab": {"id": tab}}}
    with pytest.raises(RenderError, match="invalid tab"):
        agent_browser.GaddiRenderer()(URL)
    assert daemon.methods == ["open"]

def test_reported_html_truncation_cannot_be_complete(daemon, monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("js_spa_shell.html"))
    daemon.overrides["html"] = lambda request: {
        "id": request["id"], "result": {"tab": {"id": 42, "owned": True, "url": daemon.final_url, "title": "fake", "active": True, "windowId": 1, "index": 0}, "url": URL, "html": daemon.html, "truncated": True},
    }
    # Even a fixture whose headings and word count all pass cannot override the
    # browser's independently measured capture gap.
    result = _reader(tmp_path).read(URL, clock=CLOCK, as_me=True)
    assert result.content
    assert result.completeness.status == "incomplete"
    assert any(e.name == "capture_truncation" and e.passed is False for e in result.completeness.evidence)
    assert daemon.methods.count("html") == 1


@pytest.mark.parametrize("status", [None, 201, 403])
def test_optional_status_is_honest(daemon, tmp_path, status):
    daemon.status = status
    result = _reader(tmp_path).read(URL, clock=CLOCK, as_me=True)
    assert result.provenance.http_status == status
    if status == 403:
        assert result.completeness.status == "unknown"
    else:
        assert result.completeness.status == "complete"


def test_as_me_cli_routes_through_chrome(daemon, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OMNIREAD_PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(ladder, "fetch_static", lambda _: pytest.fail("Anonymous fetch"))
    assert cli.main(["read", URL, "--as-me", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["provenance"]["tier"] == 4
    assert result["provenance"]["engine"] == "gaddi+trafilatura"


def test_profile_backed_retry_prefers_chrome(daemon, monkeypatch, tmp_path):
    reader = _reader(tmp_path)
    reader.profiles.profile_dir(URL).mkdir(parents=True)
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("authwall_page.html"))
    monkeypatch.setattr(DefuddleRenderer, "render", lambda *args: pytest.fail("Fallback despite live Chrome"))
    assert reader.read(URL, clock=CLOCK).provenance.engine == "gaddi+trafilatura"
    assert daemon.methods == ["status", "open", "html", "close"]


def test_socket_failure_after_status_does_not_fall_back(daemon, monkeypatch, tmp_path):
    reader = _reader(tmp_path)
    reader.profiles.profile_dir(URL).mkdir(parents=True)
    daemon.overrides["html"] = lambda _: b"bad json\n"
    monkeypatch.setattr(DefuddleRenderer, "render", lambda *args: pytest.fail("Fallback after successful status"))
    with pytest.raises(RenderError):
        reader.read(URL, clock=CLOCK, as_me=True)
    assert daemon.methods == ["status", "open", "html", "close"]


@pytest.mark.parametrize("login", ["profile", "chrome"])
def test_complete_public_page_with_login_preference_never_probes(daemon, monkeypatch, tmp_path, login):
    reader = _reader(tmp_path)
    if login == "profile":
        reader.profiles.profile_dir(URL).mkdir(parents=True)
    else:
        reader.profiles.remember_chrome_login(URL)
    monkeypatch.setattr(ladder, "fetch_static", lambda _: _page("normal_article.html"))
    monkeypatch.setattr(agent_browser, "available", lambda: pytest.fail("Complete public page probed Chrome"))
    result = reader.read(URL, clock=CLOCK)
    assert result.completeness.status == "complete"
    assert result.provenance.tier == 1
    assert daemon.requests == []


def test_chrome_preference_is_not_a_fallback_profile(monkeypatch, tmp_path):
    reader = _reader(tmp_path)
    reader.profiles.remember_chrome_login(URL)
    monkeypatch.setattr(DefuddleRenderer, "render", lambda *args: pytest.fail("Preference used as a profile"))
    result = reader.read(URL, clock=CLOCK, as_me=True)
    assert result.completeness.status == "unknown"
    assert result.content == ""
    assert "omniread login" in result.completeness.reason
