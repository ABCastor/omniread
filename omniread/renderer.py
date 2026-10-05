"""Tier-2 rendering through an operator-installed Playwright/Chromium bundle."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Literal

from .fetch import FetchedPage, detect_block_page
from .types import RenderError

_DEFAULT_RENDER_SCRIPT = Path(__file__).with_name("browser_render.mjs")


@dataclass(frozen=True, slots=True)
class LoginResult:
    """Metadata returned after a user closes the headed login browser."""

    final_url: str
    completion: Literal["window-closed", "timeout"]
    elapsed_ms: int


class DefuddleRenderer:
    """Thin adapter over Playwright using its bundled Chromium only.

    ``playwright_root`` is the directory containing the bundle's ``package.json``
    and ``node_modules``. No Chrome channel or system executable path is accepted.
    Optional unpacked extension paths are passed to the isolated browser context;
    OmniRead never bundles their code or licenses.
    """

    def __init__(
        self,
        *,
        script_path: Path | None = None,
        playwright_root: Path | None = None,
        extensions: Iterable[Path] = (),
        user_data_dir: Path | None = None,
        timeout_seconds: int = 70,
    ) -> None:
        self.script_path = script_path or _DEFAULT_RENDER_SCRIPT
        self.playwright_root = playwright_root or _find_defuddle_bundle()
        self.extensions = tuple(extensions)
        self.user_data_dir = user_data_dir
        self.timeout_seconds = timeout_seconds

    def render(self, url: str) -> FetchedPage:
        """Render one URL and translate browser facts into the shared fetch shape."""

        tier_label = "Tier-4" if self.user_data_dir else "Tier-2"
        completed = self._run(url, timeout=self.timeout_seconds)
        try:
            payload = json.loads(completed.stdout)
            raw_html = str(payload["html"])
            status = int(payload["status"])
            final_url = str(payload["final_url"])
            elapsed_ms = payload.get("elapsed_ms")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RenderError(
                f"{tier_label} renderer returned an invalid metadata envelope"
            ) from exc
        if not raw_html.strip():
            raise RenderError(f"{tier_label} renderer returned an empty document")
        if not 100 <= status <= 599:
            raise RenderError(f"{tier_label} renderer returned invalid HTTP status {status}")
        return FetchedPage(
            raw_html=raw_html,
            http_status=status,
            final_url=final_url,
            headers={
                "content-type": "text/html",
                "x-omniread-rendered": "true",
                "x-omniread-render-tier": "4" if self.user_data_dir else "2",
                **(
                    {"x-omniread-persistent-profile": "true"}
                    if self.user_data_dir
                    else {}
                ),
                **(
                    {"x-omniread-render-ms": str(int(elapsed_ms))}
                    if isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0
                    else {}
                ),
            },
            block_signal=detect_block_page(
                raw_html,
                http_status=status,
                final_url=final_url,
            ),
            source_urls=tuple(dict.fromkeys((url, final_url))),
        )

    def login(self, url: str, *, login_timeout_seconds: int = 600) -> LoginResult:
        """Open a headed persistent context and wait for the user to close it."""

        if self.user_data_dir is None:
            raise ValueError("Headed login requires a persistent user_data_dir")
        if isinstance(login_timeout_seconds, bool) or login_timeout_seconds <= 0:
            raise ValueError("Login timeout must be a positive number of seconds")
        self.user_data_dir.parent.mkdir(parents=True, exist_ok=True)
        completed = self._run(
            url,
            extra=("--login", "--login-timeout", str(login_timeout_seconds)),
            timeout=login_timeout_seconds + 90,
        )
        try:
            payload = json.loads(completed.stdout)
            final_url = str(payload["final_url"])
            completion = str(payload["completion"])
            elapsed_ms = int(payload["elapsed_ms"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RenderError("Tier-4 login browser returned invalid metadata") from exc
        if completion not in {"window-closed", "timeout"}:
            raise RenderError(f"Tier-4 login browser returned unknown completion {completion!r}")
        return LoginResult(
            final_url=final_url,
            completion=completion,
            elapsed_ms=elapsed_ms,
        )

    def _run(
        self,
        url: str,
        *,
        extra: tuple[str, ...] = (),
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        command = self._command(url)
        command.extend(extra)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RenderError(f"Playwright Chromium operation failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit {completed.returncode}"
            raise RenderError(f"Playwright Chromium operation failed: {detail}")
        return completed

    def _command(self, url: str) -> list[str]:
        if not self.script_path.is_file():
            raise RenderError(f"Browser render adapter is missing: {self.script_path}")
        package_json = self.playwright_root / "package.json"
        if not package_json.is_file():
            raise RenderError(
                "The Playwright bundle was not found. Set "
                "OMNIREAD_PLAYWRIGHT_ROOT to a directory with Playwright installed."
            )
        node = shutil.which("node")
        if node is None:
            raise RenderError("Browser rendering requires the Node.js executable")
        command = [
            "node" if Path(node).name == "node" else node,
            str(self.script_path),
            url,
            "--playwright-root",
            str(self.playwright_root),
        ]
        if self.user_data_dir is not None:
            command.extend(("--user-data-dir", str(self.user_data_dir)))
        for extension in self.extensions:
            command.extend(("--extension", str(extension)))
        return command


def render_page(url: str) -> FetchedPage:
    """Render with the operator-configured bundle discovery."""

    extension_paths = tuple(
        Path(value).expanduser()
        for value in os.environ.get("OMNIREAD_BROWSER_EXTENSIONS", "").split(os.pathsep)
        if value
    )
    return DefuddleRenderer(extensions=extension_paths).render(url)


def _find_defuddle_bundle() -> Path:
    configured = os.environ.get("OMNIREAD_PLAYWRIGHT_ROOT") or os.environ.get(
        "OMNIREAD_DEFUDDLE_PLAYWRIGHT_ROOT"
    )
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local/share/omniread/browser"
