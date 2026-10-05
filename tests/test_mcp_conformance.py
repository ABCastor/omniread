from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

from omniread import cli, ladder
from omniread.fetch import FetchedPage, detect_block_page
from omniread.mcp_adapter import McpAdapter
from omniread.reader import Reader
from omniread.recipes import RecipeRegistry
from omniread.types import Completeness, CostToComplete, Outline, Provenance, ReadResult, Section


FIXED_TIME = datetime(2026, 7, 12, 16, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"


def _result(*, full: bool) -> ReadResult:
    sections = [
        Section("Head", 1, "head", 10, 20),
        Section("Tail", 2, "tail", 20, 40),
    ]
    return ReadResult(
        url="https://example.test/tail-heavy",
        content="# Head\n\nCore" if not full else "# Head\n\nCore\n\n## Tail\n\nRemainder",
        outline=Outline(sections, 30, False, False, False),
        completeness=Completeness("complete", [], "Independent fixture manifest agrees"),
        provenance=Provenance(
            tier=1,
            engine="fixture-engine",
            recipe="generic",
            canonical_url="https://example.test/tail-heavy",
            fetched_at=FIXED_TIME.isoformat(),
            http_status=200,
            final_url="https://example.test/tail-heavy",
            source_urls=["https://example.test/tail-heavy"],
        ),
        structured_data=None,
        coverage="full" if full else "core",
        cost_to_complete=None if full else CostToComplete(1, 20),
        truncated=not full,
        omitted=[] if full else ["tail"],
    )


class StubReader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def read(self, url: str, *, budget, full, clock, render=None) -> ReadResult:
        self.calls.append({"url": url, "budget": budget, "full": full})
        return _result(full=full)


def _page(name: str) -> FetchedPage:
    raw_html = (FIXTURES / name).read_text(encoding="utf-8")
    return FetchedPage(
        raw_html=raw_html,
        http_status=200,
        final_url="https://example.test/final",
        headers={"content-type": "text/html"},
        block_signal=detect_block_page(raw_html, http_status=200),
    )


def test_cli_and_mcp_share_the_identical_readresult_contract(
    monkeypatch, capsys
) -> None:
    reader = StubReader()
    monkeypatch.setattr(cli, "read", reader.read)
    assert cli.main(
        ["https://example.test/tail-heavy", "--json", "--budget", "10"]
    ) == 0
    cli_payload = json.loads(capsys.readouterr().out)

    mcp = McpAdapter(reader=reader, clock=lambda: FIXED_TIME)
    mcp_payload = mcp.read_url(
        "https://example.test/tail-heavy",
        budget_tokens=10,
    )

    assert mcp_payload["result"] == cli_payload
    assert mcp_payload["result"]["coverage"] == "core"
    assert mcp_payload["result"]["completeness"]["status"] == "complete"


def test_cli_and_mcp_are_identical_through_the_real_recipe_routed_core(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("normal_article.html"))
    reader = Reader(
        registry=RecipeRegistry.discover(include_entry_points=False, local_paths=())
    )
    monkeypatch.setattr(cli, "read", reader.read)
    monkeypatch.setattr(cli, "_utc_now", lambda: FIXED_TIME)
    assert cli.main(["https://example.test/article", "--json", "--full"]) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    mcp_payload = McpAdapter(reader=reader, clock=lambda: FIXED_TIME).read_url(
        "https://example.test/article",
        full=True,
    )

    assert mcp_payload["result"] == cli_payload
    assert cli_payload["provenance"]["recipe"] == "generic"


def test_mcp_read_more_and_read_section_escalate_real_core_result(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "fetch_static", lambda url: _page("tail_heavy.html"))
    reader = Reader(
        registry=RecipeRegistry.discover(include_entry_points=False, local_paths=())
    )
    full_result = reader.read(
        "https://example.test/tail-heavy",
        full=True,
        clock=lambda: FIXED_TIME,
    )
    budget = full_result.outline.sections[0].token_count
    mcp = McpAdapter(reader=reader, clock=lambda: FIXED_TIME)
    initial = mcp.read_url("https://example.test/tail-heavy", budget_tokens=budget)

    expanded = mcp.read_more(initial["handle"])
    section = mcp.read_section(initial["handle"], "closing-checklist")

    assert initial["result"]["coverage"] == "core"
    assert initial["result"]["completeness"]["status"] == "complete"
    assert expanded["result"]["coverage"] == "full"
    assert expanded["result"]["cost_to_complete"] is None
    assert "Closing checklist" in expanded["result"]["content"]
    assert section["anchor"] == "closing-checklist"
    assert "Verify the source first" in section["content"]


def test_every_mcp_tool_ships_a_real_description() -> None:
    """The descriptions ARE the product surface, and they silently regressed once.

    FastMCP takes each tool's description from its docstring. Until 2026-07-28 all
    three functions had none, so a live `tools/list` returned `description: ''` and an
    agent choosing between OmniRead and its own built-in fetch had nothing to go on.
    A registered server nobody selects is the same as no server.
    """

    from omniread.mcp_adapter import create_server

    server = create_server(adapter=McpAdapter(reader=None, clock=lambda: FIXED_TIME))
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"read_url", "read_section", "read_more"}
    for tool in tools:
        description = (tool.description or "").strip()
        assert description, f"MCP tool {tool.name} ships no description"
        assert len(description) >= 80, (
            f"MCP tool {tool.name} description is too thin to guide tool selection"
        )

    read_url = next(tool for tool in tools if tool.name == "read_url")
    # An agent must learn the honesty contract from the tool itself.
    for term in ("complete", "incomplete", "unknown", "abstract_only"):
        assert term in read_url.description
