"""Thin MCP peer adapter over the same recipe-routed library used by the CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import secrets

from .reader import read
from .ladder import read_html
from .tokens import section_by_anchor
from .types import DependencyError, ReadResult


@dataclass(frozen=True, slots=True)
class _StoredRequest:
    url: str
    html: str | None = None
    captured_at: str | None = None


class McpAdapter:
    """Stateful handle adapter implementing ``read_url``, ``read_section``, and ``read_more``."""

    def __init__(self, *, reader=None, clock) -> None:
        self.reader = reader or read
        self.clock = clock
        self._requests: dict[str, _StoredRequest] = {}

    def read_url(
        self,
        url: str,
        *,
        budget_tokens: int | None = None,
        full: bool = False,
    ) -> dict[str, object]:
        """Return a handle plus the unmodified shared ``ReadResult`` contract."""

        result = self._read(url, budget=None if full else budget_tokens, full=full)
        handle = secrets.token_urlsafe(18)
        self._requests[handle] = _StoredRequest(url=url)
        return {"handle": handle, "result": asdict(result)}

    def extract_html(
        self, html: str, url: str, *, budget_tokens: int | None = None, full: bool = False,
    ) -> dict[str, object]:
        """Extract a caller-owned HTML snapshot and retain it for handle follow-ups."""

        result = read_html(html, url, budget=budget_tokens, full=full, clock=self.clock)
        handle = secrets.token_urlsafe(18)
        self._requests[handle] = _StoredRequest(url=url, html=html, captured_at=result.provenance.fetched_at)
        return {"handle": handle, "result": asdict(result)}

    def read_more(self, handle: str) -> dict[str, object]:
        """Escalate a prior bounded request to full coverage."""

        stored = self._stored(handle)
        result = self._read_stored(stored)
        return {"handle": handle, "result": asdict(result)}

    def read_section(self, handle: str, anchor: str) -> dict[str, object]:
        """Return one complete section from the full result behind a handle."""

        stored = self._stored(handle)
        result = self._read_stored(stored)
        try:
            content = section_by_anchor(result.content, anchor)
        except KeyError as exc:
            raise ValueError(f"Unknown section anchor {anchor!r}") from exc
        section = next(item for item in result.outline.sections if item.anchor == anchor)
        return {
            "handle": handle,
            "anchor": anchor,
            "content": content,
            "token_count": section.token_count,
            "completeness": asdict(result.completeness),
            "provenance": asdict(result.provenance),
        }

    def _read(self, url: str, *, budget: int | None, full: bool) -> ReadResult:
        read_call = self.reader.read if hasattr(self.reader, "read") else self.reader
        return read_call(url, budget=budget, full=full, clock=self.clock)

    def _read_stored(self, stored: _StoredRequest) -> ReadResult:
        if stored.html is not None:
            return read_html(stored.html, stored.url, full=True, clock=lambda: stored.captured_at)
        return self._read(stored.url, budget=None, full=True)

    def _stored(self, handle: str) -> _StoredRequest:
        try:
            return self._requests[handle]
        except KeyError as exc:
            raise ValueError(f"Unknown or expired read handle {handle!r}") from exc


def create_server(*, adapter: McpAdapter):
    """Bind the adapter to the official MCP Python SDK when that optional dep exists."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise DependencyError("The MCP server requires the official `mcp` Python SDK") from exc

    server = FastMCP("omniread")

    @server.tool()
    def read_url(url: str, budget_tokens: int | None = None, full: bool = False):
        """Read a web page and get told, with evidence, whether the read was complete.

        Prefer this over a plain fetch whenever being wrong about the content would
        matter: research, paywalled or paginated sources, academic papers, long
        threads, JavaScript-heavy pages.

        Every result carries `completeness.status`, and you should act on it:

        - `complete` means the available independent signals support a complete
          capture; it does not guarantee that the server exposed every section. Evidence comes from outside the extraction engine, so this is not
          the extractor grading its own work.
        - `incomplete` means a gap was measured, not suspected. `cost_to_complete`
          says how many items remain and roughly what retrieving them would cost.
        - `unknown` means the evidence was too thin to certify. Block pages and bot
          challenges always land here rather than passing as a successful read.

        For academic papers the result also carries
        `structured_data.scholarly.content_level`. Treat `abstract_only` as a hard
        stop: you are holding an abstract, not the paper, and must not summarize,
        quote, or characterize the paper's methods or findings as if you had read
        it. `structured_data.scholarly.unlock_path` names what would actually
        retrieve the full text.

        `budget_tokens` truncates on section boundaries and names what it omitted,
        so a bounded read stays honest about being bounded.
        """

        return adapter.read_url(url, budget_tokens=budget_tokens, full=full)

    @server.tool()
    def read_section(handle: str, anchor: str):
        """Fetch one section of a page you already read, by its outline anchor.

        Use the handle returned by `read_url` and an anchor from `outline.sections`.
        This is how a large document costs you its outline instead of its whole body:
        read once, look at the section list and token counts, then pull only the
        sections you actually need.
        """

        return adapter.read_section(handle, anchor)

    @server.tool()
    def read_more(handle: str):
        """Escalate an earlier bounded read to full coverage.

        Use this when a `read_url` result came back with `coverage: core`, meaning it
        was deliberately truncated to a token budget, and the omitted sections turn
        out to matter. Takes the handle from that read.
        """

        return adapter.read_more(handle)

    return server
