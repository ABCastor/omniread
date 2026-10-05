from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from omniread import cli
from omniread.renderer import LoginResult
from omniread.types import Completeness, Outline, Provenance, ReadResult, Section


def _result() -> ReadResult:
    return ReadResult(
        url="https://example.test/article",
        content="FULL CONTENT IS ONLY IN JSON MODE",
        outline=Outline(
            sections=[
                Section(
                    heading="Article",
                    level=1,
                    anchor="article",
                    token_count=12,
                    char_count=36,
                )
            ],
            total_token_count=12,
            has_tables=False,
            has_code=False,
            has_pagination=False,
        ),
        completeness=Completeness(
            status="complete",
            evidence=[],
            reason="Two independent signals agree",
        ),
        provenance=Provenance(
            tier=1,
            engine="test-engine",
            recipe="generic",
            canonical_url="https://example.test/article",
            fetched_at="2026-07-12T13:30:00+00:00",
            http_status=200,
            final_url="https://example.test/article",
            source_urls=["https://example.test/article"],
        ),
        structured_data={"@type": "Article"},
        coverage="full",
        cost_to_complete=None,
        truncated=False,
        omitted=[],
    )


def test_json_cli_emits_full_typed_contract(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def fake_read(url: str, *, budget, full, clock, render=None):
        observed.update(url=url, budget=budget, full=full, clock_value=clock())
        return _result()

    monkeypatch.setattr(cli, "read", fake_read)

    assert cli.main(["https://example.test/article", "--json", "--budget", "50"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["content"] == "FULL CONTENT IS ONLY IN JSON MODE"
    assert payload["completeness"]["status"] == "complete"
    assert payload["provenance"]["tier"] == 1
    assert observed["budget"] == 50
    assert observed["full"] is False
    assert isinstance(observed["clock_value"], datetime)
    assert observed["clock_value"].tzinfo is timezone.utc


def test_human_cli_is_outline_first(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "read",
        lambda url, *, budget, full, clock, render=None: _result(),
    )

    assert cli.main(["https://example.test/article"]) == 0
    output = capsys.readouterr().out

    assert "Completeness: complete" in output
    assert "Coverage: full" in output
    assert "# Outline" in output
    assert "[#article]" in output
    assert "FULL CONTENT IS ONLY IN JSON MODE" not in output


def test_full_flag_reaches_the_core(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def fake_read(url: str, *, budget, full, clock, render=None):
        observed.update(budget=budget, full=full)
        return _result()

    monkeypatch.setattr(cli, "read", fake_read)

    assert cli.main(["https://example.test/article", "--json", "--full"]) == 0
    capsys.readouterr()
    assert observed == {"budget": None, "full": True}


def test_explicit_read_subcommand_and_bare_url_are_both_supported(
    monkeypatch, capsys
) -> None:
    calls: list[str] = []

    def fake_read(url: str, *, budget, full, clock, render=None):
        calls.append(url)
        return _result()

    monkeypatch.setattr(cli, "read", fake_read)
    assert cli.main(["read", "https://example.test/article", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["https://example.test/article", "--json"]) == 0
    capsys.readouterr()
    assert calls == ["https://example.test/article", "https://example.test/article"]


def test_login_subcommand_never_accepts_or_reads_credentials(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("OMNIREAD_PROFILE_DIR", str(tmp_path / "profiles"))
    observed: dict[str, object] = {}

    class CredentialTrap:
        def read(self, *args, **kwargs):
            raise AssertionError("login must not read credentials from stdin")

        def readline(self, *args, **kwargs):
            raise AssertionError("login must not read credentials from stdin")

    class FakeRenderer:
        def __init__(self, **kwargs):
            observed["renderer_kwargs"] = kwargs

        def login(self, *args, **kwargs):
            observed.update(login_args=args, login_kwargs=kwargs)
            return LoginResult(str(args[0]), "window-closed", 1000)

    monkeypatch.setattr(sys, "stdin", CredentialTrap())
    monkeypatch.setattr(cli, "DefuddleRenderer", FakeRenderer)

    assert cli.main(["login", "news.example.com", "--login-timeout", "45"]) == 0
    output = capsys.readouterr().out

    assert observed["renderer_kwargs"] == {
        "user_data_dir": tmp_path / "profiles" / "example.com"
    }
    assert observed["login_args"] == ("https://news.example.com/",)
    assert observed["login_kwargs"] == {"login_timeout_seconds": 45}
    assert "Login was not verified" in output
    assert "The next read will confirm" in output
