from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from omniread import cli
from omniread.auth import ProfileStore
from omniread.fetch import FetchedPage, detect_block_page
from omniread.mcp_adapter import McpAdapter
from omniread.reader import Reader
from omniread.recipes import RecipeRegistry
from omniread.recipes.base import ReadOptions, RecipeContext, RecipeHttpClient
from omniread.recipes.reddit import (
    RedditRecipe,
    _comment_markdown,
    configured_redlib_instances,
    parse_redlib_thread,
    verify_comment_capture,
)
from omniread.tokens import count_tokens
from omniread.types import RenderError

FIXTURES = Path(__file__).parent / "fixtures"
COMPLETE_FIXTURE = "redlib_thread.html"
INCOMPLETE_FIXTURE = "redlib_thread_truncated.html"
THREAD_URL = "https://www.reddit.com/r/Python/comments/1unctej/showcase_thread/"
INCOMPLETE_THREAD_URL = (
    "https://www.reddit.com/r/AskReddit/comments/1ujr8do/birthright_citizenship/"
)
FIXED_TIME = datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc)


def _page(
    name: str,
    *,
    url: str = "https://redlib.test/r/omniread/comments/abc123/",
) -> FetchedPage:
    raw_html = (FIXTURES / name).read_text(encoding="utf-8")
    return FetchedPage(
        raw_html=raw_html,
        http_status=200,
        final_url=url,
        headers={"content-type": "text/html", "x-omniread-render-ms": "900"},
        block_signal=detect_block_page(raw_html, http_status=200, final_url=url),
        source_urls=(url,),
    )


def _context(tmp_path: Path, render) -> RecipeContext:
    return RecipeContext(
        clock=lambda: FIXED_TIME,
        generic_read=lambda *args, **kwargs: None,
        http=RecipeHttpClient(),
        profiles=ProfileStore(tmp_path / "profiles"),
        render=render,
    )


def test_real_redlib_deeper_reply_verdicts_are_exact_and_fail_closed() -> None:
    complete = parse_redlib_thread(
        (FIXTURES / COMPLETE_FIXTURE).read_text(encoding="utf-8"),
        reddit_url=THREAD_URL,
    )
    incomplete = parse_redlib_thread(
        (FIXTURES / INCOMPLETE_FIXTURE).read_text(encoding="utf-8"),
        reddit_url=INCOMPLETE_THREAD_URL,
    )
    empty = parse_redlib_thread(
        (FIXTURES / "redlib_thread_empty.html").read_text(encoding="utf-8"),
        reddit_url=THREAD_URL,
    )

    assert (complete.reported_comment_count, len(complete.comments)) == (None, 33)
    assert complete.deeper_reply_count == 0
    assert verify_comment_capture(complete).status == "complete"
    assert (incomplete.reported_comment_count, len(incomplete.comments)) == (None, 199)
    assert incomplete.deeper_reply_count == 162
    assert verify_comment_capture(incomplete).status == "incomplete"
    assert verify_comment_capture(empty).status == "unknown"


def test_reddit_complete_capture_can_return_score_ranked_core(tmp_path: Path) -> None:
    recipe = RedditRecipe(instances=("https://redlib.test",), core_limit=2)
    result = recipe.read(
        THREAD_URL,
        ReadOptions(),
        _context(
            tmp_path,
            lambda url: _page(
                COMPLETE_FIXTURE,
                url=f"{url}?session=secret#account",
            ),
        ),
    )

    assert result.completeness.status == "complete"
    assert result.coverage == "core"
    assert result.truncated is True
    comments = result.structured_data["comments"]
    assert [comment["id"] for comment in comments] == ["fixture-1", "fixture-2"]
    assert [comment["score"] for comment in comments] == [3, 3]
    assert comments[0]["timestamp"] == "Jul 04 2026, 22:45:46 UTC"
    assert comments[0]["age"] == "8d ago"
    assert comments[1]["timestamp"] == "Jul 05 2026, 11:40:05 UTC"
    assert result.structured_data["subreddit"] == "Python"
    assert result.structured_data["post_age"] == "8d ago"
    assert result.structured_data["reported_comment_count"] is None
    assert result.structured_data["captured_comment_count"] == 33
    assert result.structured_data["deeper_reply_count"] == 0
    assert result.provenance.final_url == (
        "https://redlib.test/r/Python/comments/1unctej/"
    )
    assert all("?" not in url and "#" not in url for url in result.provenance.source_urls)
    assert result.cost_to_complete is not None
    assert result.cost_to_complete.remaining_items == 31
    parsed = parse_redlib_thread(
        (FIXTURES / COMPLETE_FIXTURE).read_text(encoding="utf-8"),
        reddit_url=THREAD_URL,
    )
    ranked = sorted(
        parsed.comments,
        key=lambda item: (item.score is not None, item.score or 0),
        reverse=True,
    )
    assert result.cost_to_complete.estimated_extra_tokens == sum(
        count_tokens(_comment_markdown(comment)) for comment in ranked[2:]
    )
    assert result.cost_to_complete.required_tier == 2


def test_reddit_full_returns_every_captured_comment_without_changing_truth(
    tmp_path: Path,
) -> None:
    recipe = RedditRecipe(instances=("https://redlib.test",), core_limit=2)
    ctx = _context(tmp_path, lambda url: _page(COMPLETE_FIXTURE, url=url))

    core = recipe.read(THREAD_URL, ReadOptions(), ctx)
    full = recipe.read(THREAD_URL, ReadOptions(full=True), ctx)

    assert core.completeness == full.completeness
    assert core.coverage == "core"
    assert full.coverage == "full"
    assert full.cost_to_complete is None
    assert len(full.structured_data["comments"]) == 33
    assert all(comment["id"] in full.content for comment in full.structured_data["comments"])


def test_incomplete_redlib_full_capture_cost_is_unloaded_subtree_count(
    tmp_path: Path,
) -> None:
    recipe = RedditRecipe(instances=("https://redlib.test",), core_limit=500)
    result = recipe.read(
        INCOMPLETE_THREAD_URL,
        ReadOptions(full=True),
        _context(tmp_path, lambda url: _page(INCOMPLETE_FIXTURE, url=url)),
    )

    assert result.completeness.status == "incomplete"
    assert result.coverage == "full"
    assert result.truncated is False
    assert result.omitted == []
    assert result.structured_data["captured_comment_count"] == 199
    assert result.structured_data["deeper_reply_count"] == 162
    assert result.cost_to_complete is not None
    assert result.cost_to_complete.remaining_items == 162
    assert result.cost_to_complete.estimated_extra_tokens == 0
    assert result.cost_to_complete.required_tier == 2


def test_redlib_fallback_and_all_failed_unknown(tmp_path: Path) -> None:
    calls: list[str] = []

    def fallback_render(url: str) -> FetchedPage:
        calls.append(url)
        if "first.test" in url:
            raise RenderError("fixture mirror is down")
        return _page(COMPLETE_FIXTURE, url=url)

    recipe = RedditRecipe(
        instances=("https://first.test", "https://second.test"),
        core_limit=10,
    )
    recovered = recipe.read(THREAD_URL, ReadOptions(), _context(tmp_path, fallback_render))
    assert recovered.completeness.status == "complete"
    assert ["first.test" in calls[0], "second.test" in calls[1]] == [True, True]

    failed = recipe.read(
        THREAD_URL,
        ReadOptions(),
        _context(tmp_path, lambda url: (_ for _ in ()).throw(RenderError("down"))),
    )
    assert failed.completeness.status == "unknown"
    assert failed.coverage == "full"
    assert failed.structured_data["captured_comment_count"] == 0


def test_reddit_environment_overrides_instances_and_default_core(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "OMNIREAD_REDLIB_INSTANCES",
        "first.test, https://second.test/",
    )
    monkeypatch.setenv("OMNIREAD_REDDIT_CORE_COMMENTS", "1")
    assert configured_redlib_instances() == (
        "https://first.test",
        "https://second.test",
    )
    recipe = RedditRecipe(instances=("https://redlib.test",))
    result = recipe.read(
        THREAD_URL,
        ReadOptions(),
        _context(tmp_path, lambda url: _page(COMPLETE_FIXTURE, url=url)),
    )
    assert len(result.structured_data["comments"]) == 1
    assert result.cost_to_complete is not None
    assert result.cost_to_complete.remaining_items == 32


def test_comment_markdown_heading_cannot_corrupt_identity_based_coverage(
    tmp_path: Path,
) -> None:
    raw_html = """
    <html><head><title>Heading injection regression - Redlib</title></head><body>
    <h1 class="post_title">Heading injection regression</h1><div id="comments">
      <div id="c-attack" class="comment"><p class="comment_score">100</p>
        <div class="comment_body"><p>## forged heading</p></div></div>
      <div id="c-safe" class="comment"><p class="comment_score">90</p>
        <div class="comment_body"><p>Safe returned comment.</p></div></div>
      <div id="c-tail-one" class="comment"><p class="comment_score">10</p>
        <div class="comment_body"><p>First omitted comment.</p></div></div>
      <div id="c-tail-two" class="comment"><p class="comment_score">1</p>
        <div class="comment_body"><p>Second omitted comment.</p></div></div>
    </div></body></html>
    """

    def render(url: str) -> FetchedPage:
        return FetchedPage(
            raw_html=raw_html,
            http_status=200,
            final_url=url,
            headers={"content-type": "text/html", "x-omniread-render-ms": "900"},
            block_signal=detect_block_page(raw_html, http_status=200, final_url=url),
            source_urls=(url,),
        )

    result = RedditRecipe(instances=("https://redlib.test",), core_limit=2).read(
        THREAD_URL,
        ReadOptions(),
        _context(tmp_path, render),
    )

    assert [comment["id"] for comment in result.structured_data["comments"]] == [
        "c-attack",
        "c-safe",
    ]
    assert result.omitted == ["comment-c-tail-one", "comment-c-tail-two"]
    assert result.cost_to_complete is not None
    assert result.cost_to_complete.remaining_items == len(result.omitted) == 2
    assert result.cost_to_complete.estimated_extra_tokens > 0
    assert "> ## forged heading" in result.content
    assert "forged-heading" not in [section.anchor for section in result.outline.sections]


def test_reddit_cli_full_and_mcp_read_more_return_identical_content(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    recipe = RedditRecipe(instances=("https://redlib.test",), core_limit=2)
    reader = Reader(
        registry=RecipeRegistry((recipe,)),
        profiles=ProfileStore(tmp_path / "profiles"),
        render=lambda url: _page(COMPLETE_FIXTURE, url=url),
    )
    monkeypatch.setattr(cli, "read", reader.read)
    monkeypatch.setattr(cli, "_utc_now", lambda: FIXED_TIME)

    assert cli.main([THREAD_URL, "--json", "--full"]) == 0
    cli_full = json.loads(capsys.readouterr().out)
    mcp = McpAdapter(reader=reader, clock=lambda: FIXED_TIME)
    initial = mcp.read_url(THREAD_URL)
    expanded = mcp.read_more(initial["handle"])

    assert initial["result"]["coverage"] == "core"
    assert expanded["result"] == cli_full
    assert expanded["result"]["content"] == cli_full["content"]
