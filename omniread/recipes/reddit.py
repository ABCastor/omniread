"""Reddit threads through Redlib, with capture truth separate from returned coverage."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit

from selectolax.parser import HTMLParser, Node

from .base import ReadOptions, Recipe, RecipeContext, clock_iso, regexes
from .._html import _attr, normalize_space, visible_dom_snapshot
from ..fetch import FetchedPage
from ..renderer import render_page
from ..tokens import build_outline, count_tokens
from ..types import (
    Completeness,
    CostToComplete,
    Evidence,
    PolicyError,
    Provenance,
    ReadResult,
)

DEFAULT_REDLIB_INSTANCES = (
    "https://safereddit.com",
    "https://redlib.catsarch.com",
)
DEFAULT_CORE_COMMENTS = 200
REDDIT_CORE_ENV = "OMNIREAD_REDDIT_CORE_COMMENTS"
REDLIB_INSTANCES_ENV = "OMNIREAD_REDLIB_INSTANCES"

_THREAD_RE = re.compile(
    r"^https?://(?:(?:www|old|new|np)\.)?reddit\.com/"
    r"r/(?P<subreddit>[A-Za-z0-9_]+)/comments/(?P<thread_id>[a-z0-9]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RedditComment:
    """One captured Redlib comment and its native ranking/time signals."""

    id: str
    body: str
    score: int | None
    timestamp: str | None
    age: str | None
    author: str | None


@dataclass(frozen=True, slots=True)
class RedlibThread:
    """The independently counted thread representation served by Redlib."""

    title: str
    subreddit: str
    post_age: str | None
    reported_comment_count: int | None
    deeper_reply_count: int
    comments: tuple[RedditComment, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    page: FetchedPage
    thread: RedlibThread
    completeness: Completeness


class RedditRecipe(Recipe):
    """Return a score-ranked comment core while verifying the full captured tree."""

    name = "reddit"
    priority = 130
    URL_PATTERNS = regexes(
        r"^https?://(?:(?:www|old|new|np)\.)?reddit\.com/"
        r"r/[A-Za-z0-9_]+/comments/[a-z0-9]+"
    )
    RECIPE_META = {
        "name": name,
        "domains": ["reddit.com"],
        "maturity": "experimental",
        "default_core_items": DEFAULT_CORE_COMMENTS,
        "license_note": (
            "Redlib is AGPL-3.0 as an external service; "
            "no Redlib code is linked or bundled"
        ),
        "test_urls": [
            "https://www.reddit.com/r/Python/comments/example/example_thread/"
        ],
    }

    def __init__(
        self,
        *,
        instances: tuple[str, ...] | None = None,
        core_limit: int | None = None,
    ) -> None:
        self.instances = instances
        self.core_limit = core_limit

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        renderer = ctx.render or render_page
        attempted: list[str] = []
        failures: list[Evidence] = []
        candidates: list[_Candidate] = []

        for instance in self._instances():
            target = redlib_url(url, instance)
            attempted.append(target)
            try:
                page = renderer(target)
            except Exception as exc:
                failures.append(
                    Evidence(
                        f"redlib_instance_{len(attempted)}",
                        False,
                        f"{_host(target)} render failed with {type(exc).__name__}",
                    )
                )
                continue
            if page.block_signal.detected:
                failures.append(
                    Evidence(
                        f"redlib_instance_{len(attempted)}",
                        False,
                        f"{_host(target)} returned a block/challenge representation",
                    )
                )
                continue
            if not 200 <= page.http_status < 300:
                failures.append(
                    Evidence(
                        f"redlib_instance_{len(attempted)}",
                        False,
                        f"{_host(target)} returned HTTP {page.http_status}",
                    )
                )
                continue

            try:
                thread = parse_redlib_thread(page.raw_html, reddit_url=url)
            except Exception as exc:
                failures.append(
                    Evidence(
                        f"redlib_instance_{len(attempted)}",
                        False,
                        f"{_host(target)} parse failed with {type(exc).__name__}",
                    )
                )
                continue
            completeness = verify_comment_capture(thread)
            if thread.comments:
                candidates.append(_Candidate(page, thread, completeness))
                if completeness.status == "complete":
                    return self._result(
                        url=url,
                        candidate=candidates[-1],
                        attempted=attempted,
                        opts=opts,
                        ctx=ctx,
                    )
            else:
                failures.extend(completeness.evidence)

        if candidates:
            best = max(candidates, key=_candidate_rank)
            return self._result(
                url=url,
                candidate=best,
                attempted=attempted,
                opts=opts,
                ctx=ctx,
            )
        return _unknown_result(
            url=url,
            attempted=attempted,
            failures=failures,
            ctx=ctx,
        )

    def _instances(self) -> tuple[str, ...]:
        return self.instances or configured_redlib_instances()

    def _core_limit(self) -> int:
        if self.core_limit is not None:
            value = self.core_limit
        else:
            raw = os.environ.get(REDDIT_CORE_ENV, "").strip()
            if not raw:
                value = int(self.RECIPE_META["default_core_items"])
            elif raw.isdigit():
                value = int(raw)
            else:
                raise PolicyError(f"{REDDIT_CORE_ENV} must be a positive integer")
        if isinstance(value, bool) or value <= 0:
            raise PolicyError("Reddit core comment limit must be a positive integer")
        return value

    def _result(
        self,
        *,
        url: str,
        candidate: _Candidate,
        attempted: list[str],
        opts: ReadOptions,
        ctx: RecipeContext,
    ) -> ReadResult:
        page = candidate.page
        thread = candidate.thread
        ranked = sorted(
            thread.comments,
            key=lambda item: (
                item.score is not None,
                item.score if item.score is not None else 0,
            ),
            reverse=True,
        )
        full_markdown = _thread_markdown(thread, ranked)
        outline = build_outline(full_markdown)
        desired = len(ranked) if opts.full else min(self._core_limit(), len(ranked))
        include_header, selected_count = _fit_comments(
            thread,
            ranked,
            desired=desired,
            budget=None if opts.full else opts.budget_tokens,
        )
        selected = ranked[:selected_count]
        content = _thread_markdown(thread, selected) if include_header else ""
        omitted_comments = ranked[selected_count:] if include_header else ranked
        omitted = [_comment_anchor(comment) for comment in omitted_comments]
        truncated = bool(omitted)
        latency = page.headers.get("x-omniread-render-ms")
        cost = (
            CostToComplete(
                remaining_items=len(omitted_comments) + thread.deeper_reply_count,
                estimated_extra_tokens=sum(
                    count_tokens(_comment_markdown(comment))
                    for comment in omitted_comments
                ),
                required_tier=2,
                estimated_latency_ms=(
                    int(latency) if latency and latency.isdigit() else None
                ),
            )
            if truncated or thread.deeper_reply_count > 0
            else None
        )
        structured = {
            "subreddit": thread.subreddit,
            "post_age": thread.post_age,
            "reported_comment_count": thread.reported_comment_count,
            "captured_comment_count": len(thread.comments),
            "deeper_reply_count": thread.deeper_reply_count,
            "comments": [
                {
                    "id": comment.id,
                    "author": comment.author,
                    "score": comment.score,
                    "timestamp": comment.timestamp,
                    "age": comment.age,
                }
                for comment in selected
            ],
        }
        source_urls = list(
            dict.fromkeys(
                [*attempted, *getattr(page, "source_urls", ()), page.final_url]
            )
        )
        return ReadResult(
            url=url,
            content=content,
            outline=outline,
            completeness=candidate.completeness,
            provenance=Provenance(
                tier=2,
                engine="redlib+playwright-chromium",
                recipe=self.name,
                canonical_url=url,
                fetched_at=clock_iso(ctx.clock),
                http_status=page.http_status,
                final_url=page.final_url,
                source_urls=source_urls,
            ),
            structured_data=structured,
            coverage="core" if truncated else "full",
            cost_to_complete=cost,
            truncated=truncated,
            omitted=omitted,
        )


def configured_redlib_instances() -> tuple[str, ...]:
    """Return normalized fallback instances from the comma-separated environment value."""

    raw = os.environ.get(REDLIB_INSTANCES_ENV, "").strip()
    values = raw.split(",") if raw else list(DEFAULT_REDLIB_INSTANCES)
    normalized = tuple(
        dict.fromkeys(
            _normalize_instance(value) for value in values if value.strip()
        )
    )
    if not normalized:
        raise PolicyError(f"{REDLIB_INSTANCES_ENV} did not contain a usable instance")
    return normalized


def redlib_url(reddit_url: str, instance: str) -> str:
    """Translate one supported Reddit thread URL to the same route on Redlib."""

    match = _THREAD_RE.search(reddit_url)
    if not match:
        raise ValueError(f"Not a supported Reddit thread URL: {reddit_url}")
    base = _normalize_instance(instance)
    return (
        f"{base}/r/{match.group('subreddit')}/comments/"
        f"{match.group('thread_id')}/"
    )


def parse_redlib_thread(raw_html: str, *, reddit_url: str) -> RedlibThread:
    """Parse Redlib comments and its explicit unloaded-subtree links."""

    match = _THREAD_RE.search(reddit_url)
    if not match:
        raise ValueError(f"Not a supported Reddit thread URL: {reddit_url}")
    tree = HTMLParser(raw_html)
    visible = visible_dom_snapshot(raw_html).text
    counts = [
        int(value.replace(",", ""))
        for value in re.findall(r"\b([\d,]+)\s+comments?\b", visible, re.I)
    ]
    reported = max(counts) if counts else None
    title_node = _first(tree, ".post_title", ".post-title", "h1", "title")
    title = normalize_space(title_node.text(separator=" ")) if title_node else "Reddit thread"
    title = re.sub(r"\s*[-|]\s*Redlib\s*$", "", title, flags=re.I) or "Reddit thread"
    post_time = _first(
        tree,
        ".post_time",
        ".post-time",
        ".post .created",
        ".post [class*='time']",
        "article time",
    )
    post_age = normalize_space(post_time.text(separator=" ")) if post_time else None

    comments: list[RedditComment] = []
    for index, node in enumerate(tree.css(".comment"), 1):
        body_node = _first(
            node,
            ".comment_body",
            ".comment-body",
            ".comment_content",
            ".comment-content",
            ".md",
        )
        body = normalize_space(body_node.text(separator="\n")) if body_node else ""
        if not body:
            continue
        score_node = _first(node, ".comment_score", ".comment-score", "[class*='score']")
        time_node = _first(
            node,
            "time",
            ".comment_time",
            ".comment-time",
            ".created",
            "[class*='timestamp']",
        )
        author_node = _first(
            node,
            ".comment_author",
            ".comment-author",
            "[class*='author']",
        )
        comment_id = _attr(node, "id") or _attr(node, "data-id") or f"captured-{index}"
        time_text = normalize_space(time_node.text(separator=" ")) if time_node else ""
        timestamp = None
        if time_node is not None:
            timestamp = _attr(time_node, "datetime") or _attr(time_node, "title") or None
        comments.append(
            RedditComment(
                id=comment_id,
                body=body,
                score=_parse_score(score_node.text(separator=" ") if score_node else ""),
                timestamp=timestamp,
                age=time_text or None,
                author=(
                    normalize_space(author_node.text(separator=" "))
                    if author_node is not None
                    else None
                ),
            )
        )
    return RedlibThread(
        title=title,
        subreddit=match.group("subreddit"),
        post_age=post_age,
        reported_comment_count=reported,
        deeper_reply_count=len(tree.css("a.deeper_replies")),
        comments=tuple(comments),
    )


def verify_comment_capture(thread: RedlibThread) -> Completeness:
    """Judge Redlib capture only; deliberate top-N return coverage is irrelevant here."""

    captured = len(thread.comments)
    deeper = thread.deeper_reply_count
    if captured == 0:
        evidence = Evidence(
            "redlib_deeper_replies",
            None,
            f"Parsed no comment bodies; found {deeper} unloaded-subtree links",
        )
        return Completeness(
            "unknown",
            [evidence],
            "Nothing was parsed, so comment-tree completeness cannot be judged",
        )
    if deeper > 0:
        evidence = Evidence(
            "redlib_deeper_replies",
            False,
            f"Captured {captured} comments; {deeper} deeper-reply links mark unloaded subtrees",
        )
        return Completeness(
            "incomplete",
            [evidence],
            f"Redlib left {deeper} comment subtrees unloaded",
        )
    if thread.reported_comment_count is not None and captured < thread.reported_comment_count:
        return Completeness(
            "incomplete",
            [Evidence(
                "redlib_comment_count", False,
                f"Captured {captured} comments against {thread.reported_comment_count} declared comments",
            )],
            "The declared comment count exceeds the captured comment bodies",
        )
    evidence = Evidence(
        "redlib_deeper_replies",
        True,
        f"Captured {captured} comments and found no unloaded-subtree links",
    )
    return Completeness(
        "complete",
        [evidence],
        "Every visible Redlib comment subtree was inlined and parsed",
    )


def _unknown_result(
    *,
    url: str,
    attempted: list[str],
    failures: list[Evidence],
    ctx: RecipeContext,
) -> ReadResult:
    evidence = failures or [
        Evidence("redlib_instances", None, "No Redlib instance produced parseable comments")
    ]
    final_url = attempted[-1] if attempted else url
    match = _THREAD_RE.search(url)
    return ReadResult(
        url=url,
        content="",
        outline=build_outline(""),
        completeness=Completeness(
            "unknown",
            evidence,
            "Every configured Redlib instance failed or returned no parseable comment tree",
        ),
        provenance=Provenance(
            tier=2,
            engine="redlib+playwright-chromium",
            recipe="reddit",
            canonical_url=url,
            fetched_at=clock_iso(ctx.clock),
            http_status=None,
            final_url=final_url,
            source_urls=attempted or [url],
        ),
        structured_data={
            "subreddit": match.group("subreddit") if match else None,
            "post_age": None,
            "reported_comment_count": None,
            "captured_comment_count": 0,
            "deeper_reply_count": None,
            "comments": [],
        },
        coverage="full",
        cost_to_complete=None,
        truncated=False,
        omitted=[],
    )


def _thread_markdown(thread: RedlibThread, comments: list[RedditComment]) -> str:
    rendered = [_thread_header_markdown(thread)]
    rendered.extend(_comment_markdown(comment) for comment in comments)
    return "\n\n".join(rendered)


def _thread_header_markdown(thread: RedlibThread) -> str:
    header = [
        f"# {thread.title}",
        f"Subreddit: r/{thread.subreddit}",
        (
            f"Reported comments: {thread.reported_comment_count}"
            if thread.reported_comment_count is not None
            else "Reported comments: unknown"
        ),
        f"Unloaded reply subtrees: {thread.deeper_reply_count}",
    ]
    if thread.post_age:
        header.append(f"Post age: {thread.post_age}")
    return "\n\n".join(header)


def _comment_markdown(comment: RedditComment) -> str:
    metadata = [
        f"Score: {comment.score if comment.score is not None else 'unknown'}",
    ]
    if comment.author:
        metadata.append(f"Author: {comment.author}")
    if comment.timestamp:
        metadata.append(f"Timestamp: {comment.timestamp}")
    if comment.age:
        metadata.append(f"Age: {comment.age}")
    body = "\n".join(
        f"> {line}" if line else ">" for line in comment.body.splitlines()
    )
    return f"## Comment {comment.id}\n\n" + " | ".join(metadata) + f"\n\n{body}"


def _fit_comments(
    thread: RedlibThread,
    comments: list[RedditComment],
    *,
    desired: int,
    budget: int | None,
) -> tuple[bool, int]:
    if budget is None:
        return True, desired
    if isinstance(budget, bool) or budget < 0:
        raise PolicyError("Token budget must be a non-negative integer")
    header_tokens = count_tokens(_thread_header_markdown(thread))
    if header_tokens > budget:
        return False, 0
    used = header_tokens
    selected = 0
    for comment in comments[:desired]:
        comment_tokens = count_tokens(_comment_markdown(comment))
        if used + comment_tokens > budget:
            break
        used += comment_tokens
        selected += 1
    return True, selected


def _comment_anchor(comment: RedditComment) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", comment.id.lower()).strip("-")
    return f"comment-{normalized or 'captured'}"


def _candidate_rank(candidate: _Candidate) -> tuple[int, int, int]:
    status_rank = {"unknown": 0, "incomplete": 1, "complete": 2}[
        candidate.completeness.status
    ]
    captured = len(candidate.thread.comments)
    return status_rank, -candidate.thread.deeper_reply_count, captured


def _normalize_instance(value: str) -> str:
    candidate = value.strip().rstrip("/")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise PolicyError(f"Invalid Redlib instance: {value!r}")
    if parsed.username is not None or parsed.password is not None:
        raise PolicyError("Redlib instance URLs must not contain credentials")
    return candidate


def _host(url: str) -> str:
    return urlsplit(url).hostname or "configured Redlib instance"


def _first(node: HTMLParser | Node, *selectors: str) -> Node | None:
    for selector in selectors:
        found = node.css_first(selector)
        if found is not None:
            return found
    return None


def _parse_score(value: str) -> int | None:
    match = re.search(r"([-+]?\d[\d,.]*)([kKmM]?)", value)
    if not match:
        return None
    number = float(match.group(1).replace(",", ""))
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2).lower()]
    return int(round(number * multiplier))
