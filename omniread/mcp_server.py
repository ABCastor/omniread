"""Executable stdio MCP server backed by the official SDK."""

from __future__ import annotations

from datetime import datetime, timezone
import sys

from .types import DependencyError

from .mcp_adapter import McpAdapter, create_server


def main() -> None:
    """Run OmniRead's fixed three-tool MCP surface over stdio."""

    adapter = McpAdapter(reader=None, clock=lambda: datetime.now(timezone.utc))
    try:
        create_server(adapter=adapter).run()
    except DependencyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
