"""HTML extraction with the same handle/result envelope as MCP ``read_url``."""

from datetime import datetime, timezone

from .mcp_adapter import McpAdapter

_adapter = McpAdapter(clock=lambda: datetime.now(timezone.utc))


def extract_html(
    html: str, url: str, budget_tokens: int | None = None, full: bool = False,
) -> dict[str, object]:
    """Extract without network access; handles live in this Python process.

    Follow up through this module's ``read_section`` and ``read_more``. Embedded
    MCP users can instead use all three methods on their own ``McpAdapter``.
    """

    return _adapter.extract_html(html, url, budget_tokens=budget_tokens, full=full)


def read_section(handle: str, anchor: str) -> dict[str, object]:
    """Read a complete section from the retained HTML snapshot."""

    return _adapter.read_section(handle, anchor)


def read_more(handle: str) -> dict[str, object]:
    """Return the full retained HTML extraction, without refetching its URL."""

    return _adapter.read_more(handle)
