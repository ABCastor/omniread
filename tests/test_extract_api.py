from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniread import cli, extract_api, ladder
from omniread.fetch import FetchedPage, detect_block_page
from omniread.mcp_adapter import McpAdapter
from omniread.reader import Reader
from omniread.recipes import RecipeRegistry
from omniread.recipes.generic import GenericRecipe
from omniread.types import BudgetError, ExtractionError, PolicyError

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = json.loads((FIXTURES / "corpus.json").read_text())
CASES = [case for case in CORPUS["cases"] if not case.get("recipe")]
URL = "https://example.test/page"
CLOCK = lambda: "2026-09-12T12:00:00+00:00"


def _no_network(*args, **kwargs):
    pytest.fail("Supplied HTML and its handles must never fetch or render")


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_gold_cli_html_matches_url_markdown_and_verdict(case, monkeypatch, capsys):
    path = FIXTURES / case["file"]
    html = path.read_text()
    page = FetchedPage(html, 200, URL, {}, detect_block_page(html, http_status=200))
    monkeypatch.setattr(ladder, "fetch_static", lambda _: page)
    # The URL comparison must see exactly the same representation, including shells.
    reader = Reader(registry=RecipeRegistry((GenericRecipe(),)), render=lambda _: page)
    if case["name"] == "js_spa_shell":
        # URL reads escalate then raise when no content arrives. Snapshot reads
        # cannot escalate, so they must report the empty representation as unknown.
        with pytest.raises(ExtractionError):
            reader.read(URL, full=True, clock=CLOCK)
        expected_content = ""
    else:
        expected_content = reader.read(URL, full=True, clock=CLOCK).content
    monkeypatch.setattr(ladder, "fetch_static", _no_network)
    monkeypatch.setattr(ladder, "render_page", _no_network)
    assert cli.main(["extract", "--html", str(path), "--url", URL, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    result = payload["result"]
    assert payload["handle"]
    assert result["content"] == expected_content
    assert result["completeness"]["status"] == case["expected_status"]
    assert result["provenance"]["engine"] == "supplied-html+trafilatura"
    assert result["provenance"]["http_status"] is None
    assert result["provenance"]["tier"] is None


def test_snapshot_handles_recover_omitted_sections_without_fetching(monkeypatch):
    monkeypatch.setattr(ladder, "fetch_static", _no_network)
    monkeypatch.setattr(ladder, "render_page", _no_network)
    html = (FIXTURES / "tail_heavy.html").read_text()
    full = extract_api.extract_html(html, URL, full=True)["result"]
    budget = full["outline"]["sections"][0]["token_count"]
    bounded = extract_api.extract_html(html, URL, budget_tokens=budget)
    handle, result = bounded["handle"], bounded["result"]
    assert result["coverage"] == "core"
    assert result["completeness"]["status"] == "complete"
    assert result["outline"] == full["outline"]
    assert result["omitted"]
    assert result["cost_to_complete"]["remaining_items"] == len(result["omitted"])
    section = extract_api.read_section(handle, result["omitted"][-1])
    assert section["content"] in full["content"]
    assert section["anchor"] == result["omitted"][-1]
    more = extract_api.read_more(handle)
    assert more["handle"] == handle
    assert more["result"]["content"] == full["content"]
    assert more["result"]["coverage"] == "full"
    assert not more["result"]["truncated"]


def test_embedded_adapter_shares_html_handles_with_its_section_methods():
    adapter = McpAdapter(reader=_no_network, clock=CLOCK)
    html = (FIXTURES / "docs_with_code.html").read_text()
    response = adapter.extract_html(html, URL, budget_tokens=0)
    handle = response["handle"]
    assert response["result"]["content"] == ""
    more = adapter.read_more(handle)
    assert more["result"]["content"]
    anchor = more["result"]["outline"]["sections"][-1]["anchor"]
    assert adapter.read_section(handle, anchor)["content"] in more["result"]["content"]


def test_html_handles_stay_bound_to_their_snapshot():
    adapter = McpAdapter(reader=_no_network, clock=CLOCK)
    first = adapter.extract_html((FIXTURES / "normal_article.html").read_text(), URL)
    second = adapter.extract_html((FIXTURES / "docs_with_code.html").read_text(), URL)
    assert first["handle"] != second["handle"]
    assert adapter.read_more(first["handle"])["result"] == first["result"]
    assert adapter.read_more(second["handle"])["result"] == second["result"]
    with pytest.raises(ValueError, match="Unknown section anchor"):
        adapter.read_section(first["handle"], "does-not-exist")
    with pytest.raises(ValueError, match="Unknown or expired"):
        adapter.read_more("does-not-exist")


def test_full_overrides_budget_and_plain_cli_includes_markdown(capsys):
    path = FIXTURES / "docs_with_code.html"
    assert cli.main(["extract", "--html", str(path), "--url", URL, "--budget", "0", "--full", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["coverage"] == "full"
    assert result["outline"]["has_code"]
    assert cli.main(["extract", "--html", str(path), "--url", URL]) == 0
    output = capsys.readouterr().out
    assert "Completeness: complete" in output
    assert "# Outline" in output
    assert result["content"] in output


def test_html_auth_wall_scrubs_content_and_structured_side_channel():
    result = extract_api.extract_html((FIXTURES / "authwall_page.html").read_text(), URL)["result"]
    assert result["completeness"]["status"] == "unknown"
    assert result["content"] == ""
    assert result["structured_data"] == {}


@pytest.mark.parametrize("budget", [-1, True, "5"])
def test_bad_budget_fails_with_typed_error(budget):
    with pytest.raises(BudgetError):
        extract_api.extract_html((FIXTURES / "normal_article.html").read_text(), URL, budget_tokens=budget)


def test_invalid_url_and_missing_html_file_fail_cleanly(tmp_path, capsys):
    with pytest.raises(PolicyError):
        extract_api.extract_html("<p>Hello</p>", "file:///private/page.html")
    with pytest.raises(SystemExit) as exc:
        cli.main(["extract", "--html", str(tmp_path / "missing.html"), "--url", URL])
    assert exc.value.code == 1
    assert "omniread:" in capsys.readouterr().err
