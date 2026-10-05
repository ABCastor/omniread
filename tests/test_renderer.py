from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from omniread import renderer


@dataclass
class Completed:
    stdout: str
    stderr: str = ""
    returncode: int = 0


def test_renderer_invokes_the_defuddle_playwright_bundle_not_system_chrome(
    monkeypatch, tmp_path: Path
) -> None:
    script = tmp_path / "render.mjs"
    package = tmp_path / "package.json"
    script.write_text("// fixture", encoding="utf-8")
    package.write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return Completed(
            stdout=json.dumps(
                {
                    "html": "<html><body><main>Rendered</main></body></html>",
                    "status": 200,
                    "final_url": "https://example.test/final",
                    "elapsed_ms": 1200,
                }
            )
        )

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    browser = renderer.DefuddleRenderer(
        script_path=script,
        playwright_root=tmp_path,
        extensions=(tmp_path / "cleaner",),
    )

    page = browser.render("https://example.test/start")

    command = observed["command"]
    assert command[:3] == ["node", str(script), "https://example.test/start"]
    assert "--playwright-root" in command
    assert "--extension" in command
    assert "channel" not in " ".join(command).lower()
    assert "google chrome" not in " ".join(command).lower()
    assert page.http_status == 200
    assert page.final_url == "https://example.test/final"
    assert page.headers["x-omniread-render-ms"] == "1200"
    assert page.headers["x-omniread-render-tier"] == "2"


def test_persistent_profile_marks_tier4_and_login_uses_the_same_seam(
    monkeypatch, tmp_path: Path
) -> None:
    script = tmp_path / "render.mjs"
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    script.write_text("// fixture", encoding="utf-8")
    profile = tmp_path / "profiles" / "example.com"
    commands: list[list[str]] = []
    timeouts: list[int] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        timeouts.append(kwargs["timeout"])
        if "--login" in command:
            return Completed(
                stdout=json.dumps(
                    {
                        "final_url": "https://example.com/account",
                        "completion": "window-closed",
                        "elapsed_ms": 2500,
                    }
                )
            )
        return Completed(
            stdout=json.dumps(
                {
                    "html": "<html><body><main>Authenticated</main></body></html>",
                    "status": 200,
                    "final_url": "https://example.com/private",
                    "elapsed_ms": 800,
                }
            )
        )

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    browser = renderer.DefuddleRenderer(
        script_path=script,
        playwright_root=tmp_path,
        user_data_dir=profile,
    )

    page = browser.render("https://example.com/private")
    login = browser.login("https://example.com", login_timeout_seconds=45)

    assert ["--user-data-dir", str(profile)] == commands[0][
        commands[0].index("--user-data-dir") : commands[0].index("--user-data-dir") + 2
    ]
    assert "--login" in commands[1]
    assert commands[1][commands[1].index("--login-timeout") + 1] == "45"
    assert timeouts == [70, 135]
    assert page.headers["x-omniread-render-tier"] == "4"
    assert page.headers["x-omniread-persistent-profile"] == "true"
    assert login.completion == "window-closed"
