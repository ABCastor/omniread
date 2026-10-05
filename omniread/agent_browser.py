"""Render through Gaddi's gated NDJSON Unix socket, using only the stdlib."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

from .fetch import BlockSignal, FetchedPage, _validate_http_url, detect_block_page
from .types import RenderError

_MAX_FRAME_BYTES = 32 * 1024 * 1024  # The daemon's 5 MB HTML may be JSON-escaped.


def socket_path() -> Path:
    """Resolve the explicit socket or the user broker socket."""

    configured = os.environ.get("GADDI_SOCKET")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library/Application Support/Gaddi/gaddi.sock"


class _GateError(RenderError):
    def __init__(self, error: dict) -> None:
        self.code = error["code"]
        approval = error.get("approval")
        self.approval_id = approval.get("id") if isinstance(approval, dict) else None
        detail = f"Gaddi policy {self.code}: {error.get('message', self.code)}"
        if self.approval_id:
            detail += f"; approval.id={self.approval_id}"
        super().__init__(detail)


def _rpc(path: Path, method: str, params: dict, *, deadline: float) -> dict:
    request_id = uuid4().hex

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise RenderError(f"Gaddi {method} timed out")
        return seconds

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(remaining())
            connection.connect(str(path))
            request = {"id": request_id, "method": method, "params": params}
            connection.settimeout(remaining())
            connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
            frame = bytearray()
            while b"\n" not in frame:
                connection.settimeout(remaining())
                chunk = connection.recv(min(65536, _MAX_FRAME_BYTES + 1 - len(frame)))
                if not chunk:
                    raise RenderError(f"Gaddi {method} closed without a complete response")
                frame.extend(chunk)
                if len(frame) > _MAX_FRAME_BYTES:
                    raise RenderError(f"Gaddi {method} response exceeds the frame limit")
        response = json.loads(frame.split(b"\n", 1)[0])
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise RenderError(f"Gaddi {method} returned an invalid response id")
        if "error" in response:
            error = response["error"]
            if isinstance(error, dict) and error.get("code") in {"held", "denied"}:
                raise _GateError(error)
            raise RenderError(f"Gaddi {method} failed: {error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RenderError(f"Gaddi {method} returned an invalid result")
        return result
    except (OSError, ValueError, UnicodeError) as exc:
        raise RenderError(f"Gaddi {method} failed: {exc}") from exc


def available() -> bool:
    """Return whether the selected socket exists and answers a bounded status RPC."""

    path = socket_path()
    try:
        if not path.exists():
            return False
        _rpc(path, "status", {}, deadline=time.monotonic() + 1.0)
        return True
    except (OSError, RenderError):
        return False


class GaddiRenderer:
    """One background Chrome tab per authenticated render, with the existing 70-second browser budget.

    Cookies remain in the user's Chrome. Holds and denials return block evidence; no
    approval is consumed or retried here. Tab cleanup is always attempted.
    """

    engine = "gaddi+trafilatura"

    def __init__(self, *, timeout_seconds: float = 70) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Render timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def __call__(self, url: str) -> FetchedPage:
        """Implement the ladder's ``Renderer`` callable."""

        return self.render(url)

    def render(self, url: str) -> FetchedPage:
        """Return rendered HTML and transport/block evidence, then close only the tab opened here."""

        _validate_http_url(url)
        path = socket_path()
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        tab: int | None = None
        headers = {
            "content-type": "text/html",
            "x-omniread-rendered": "true",
            "x-omniread-render-tier": "4",
            "x-omniread-render-engine": self.engine,
        }
        try:
            opened = _rpc(path, "open", {
                "url": url, "foreground": False, "caller": "omniread",
            }, deadline=deadline)
            tab = opened.get("tab")
            if not isinstance(tab, dict):
                tab = None
                raise RenderError("Gaddi returned an invalid tab")
            tab = tab.get("id")
            if type(tab) is not int or tab < 0:
                tab = None
                raise RenderError("Gaddi returned an invalid tab")
            page = _rpc(path, "html", {"tab": tab}, deadline=deadline)
            html, final_url, status = page.get("html"), page.get("url"), page.get("status")
            if not isinstance(html, str) or not html.strip():
                raise RenderError("Gaddi returned an empty or invalid HTML document")
            if not isinstance(final_url, str) or not final_url:
                raise RenderError("Gaddi returned an invalid final URL")
            _validate_http_url(final_url)
            if status is None:
                # FetchedPage uses an integer internally; never report this sentinel
                # as an observed HTTP response in the public provenance.
                status = 200
                headers["x-omniread-http-status-unknown"] = "true"
            elif type(status) is not int or not 100 <= status <= 599:
                raise RenderError("Gaddi returned an invalid HTTP status")
            if page.get("truncated") is True:
                headers["x-omniread-capture-truncated"] = "true"
            headers["x-omniread-render-ms"] = str(int((time.monotonic() - started) * 1000))
            return FetchedPage(
                raw_html=html, http_status=status, final_url=final_url, headers=headers,
                block_signal=detect_block_page(html, http_status=status, final_url=final_url),
                source_urls=tuple(dict.fromkeys((url, final_url))),
            )
        except _GateError as exc:
            # 403 is an internal policy sentinel, not an observed website response.
            headers["x-omniread-policy"] = exc.code
            return FetchedPage(
                raw_html="", http_status=403, final_url=url, headers=headers,
                block_signal=BlockSignal(True, (str(exc),)), source_urls=(url,),
            )
        finally:
            if tab is not None:
                try:
                    _rpc(path, "close", {"tabs": [tab]}, deadline=time.monotonic() + 2.0)
                except RenderError:
                    # A disconnected daemon cannot acknowledge cleanup. Never mask
                    # the original failure or turn a policy hold into another attempt.
                    pass
