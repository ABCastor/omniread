"""Thin command-line adapter over OmniRead's typed core."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

from .auth import ProfileStore, normalize_site_url
from . import agent_browser
from .extract_api import extract_html
from .ladder import read_html
from .reader import read
from .renderer import DefuddleRenderer
from .tokens import render_outline_head, tokenizer_backend
from .types import OmniReadError, ReadResult


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniread",
        description="Read a URL with an explicit, independently verified completeness verdict.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    read_parser = commands.add_parser("read", help="Read one URL")
    read_parser.add_argument("url")
    read_parser.add_argument("--as-me", action="store_true", help="Read through your own logged-in browser")
    read_parser.add_argument("--json", action="store_true", dest="as_json")
    read_parser.add_argument("--budget", type=_non_negative_int, metavar="N")
    read_parser.add_argument(
        "--full",
        action="store_true",
        help="Return the full verified result, overriding any bounded core budget.",
    )
    extract_parser = commands.add_parser("extract", help="Extract HTML already held by a browser")
    extract_parser.add_argument("--html", required=True, type=Path, metavar="FILE")
    extract_parser.add_argument("--url", required=True)
    extract_parser.add_argument("--json", action="store_true", dest="as_json")
    extract_parser.add_argument("--budget", type=_non_negative_int, metavar="N")
    extract_parser.add_argument("--full", action="store_true")
    login_parser = commands.add_parser(
        "login",
        help="Open a headed browser so you can save your own login for one site",
    )
    login_parser.add_argument("site", metavar="URL|SITE")
    login_parser.add_argument(
        "--login-timeout",
        type=_positive_int,
        default=600,
        metavar="SECONDS",
        help="Close and save the browser after this many seconds (default: 600).",
    )
    return parser


def emit(result: ReadResult, *, as_json: bool) -> None:
    """Serialize one result without changing its contract."""

    if as_json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(_human_outline(result))


def main(argv: Sequence[str] | None = None) -> int:
    """Run a read or user-performed persistent-profile login."""

    parser = _parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] not in {"read", "extract", "login", "-h", "--help"}:
        raw_args.insert(0, "read")
    args = parser.parse_args(raw_args)

    try:
        if args.command == "login":
            return _login(args.site, login_timeout=args.login_timeout)
        if args.command == "extract":
            html = args.html.read_text(encoding="utf-8")
            if args.as_json:
                print(json.dumps(extract_html(
                    html, args.url, budget_tokens=args.budget, full=args.full,
                ), ensure_ascii=False, indent=2))
            else:
                result = read_html(html, args.url, budget=args.budget, full=args.full, clock=_utc_now)
                print(_human_outline(result))
                print("\n" + result.content)
            return 0
        result = read(
            args.url,
            budget=None if args.full else args.budget,
            full=args.full,
            clock=_utc_now,
            **({"as_me": True} if args.as_me else {}),
        )
    except (OmniReadError, ValueError, TypeError, OSError) as exc:
        parser.exit(1, f"omniread: {exc}\n")

    emit(result, as_json=args.as_json)
    return 0


def _login(site: str, *, login_timeout: int) -> int:
    target_url = normalize_site_url(site)
    profiles = ProfileStore()
    domain = profiles.domain(target_url)
    if agent_browser.available():
        profiles.remember_chrome_login(target_url)
        print("Log in once in your own Chrome; OmniRead reads through it")
        return 0
    profile_dir = profiles.prepare(target_url)
    print(
        f"Opening {target_url} for {domain}. Log in yourself, then close the browser window."
    )
    print("OmniRead does not see or type your credentials.")
    outcome = DefuddleRenderer(user_data_dir=profile_dir).login(
        target_url,
        login_timeout_seconds=login_timeout,
    )
    print(f"Chromium profile directory for {domain} now exists at {profile_dir}")
    print("OmniRead never requested, typed, logged, or returned your password.")
    if outcome.completion == "timeout":
        print("The login window timed out. Login was not verified.")
    else:
        print("The browser window closed. Login was not verified.")
    print(
        "The next read will confirm whether this profile clears the login wall. "
        f"If it still returns unknown, retry `omniread login {domain}`."
    )
    return 0


def _human_outline(result: ReadResult) -> str:
    provenance = result.provenance
    lines = [
        f"URL: {result.url}",
        f"Completeness: {result.completeness.status}",
        f"Reason: {result.completeness.reason}",
        f"Coverage: {result.coverage}",
        (
            f"Provenance: tier {provenance.tier}, {provenance.engine}, "
            f"HTTP {provenance.http_status if provenance.http_status is not None else 'n/a'}"
        ),
        f"Final URL: {provenance.final_url}",
        f"Fetched at: {provenance.fetched_at}",
        f"Token counter: {tokenizer_backend()} (all counts approximate)",
    ]
    if result.truncated:
        lines.append(f"Truncated on section boundaries; omitted: {', '.join(result.omitted)}")
    if result.cost_to_complete is not None:
        cost = result.cost_to_complete
        lines.append(
            f"Cost to complete: {cost.remaining_items} items, "
            f"approximately {cost.estimated_extra_tokens} extra tokens"
        )
    notice = _notice(result.structured_data)
    if notice:
        lines.append(f"Notice: {notice}")
    lines.extend(("", render_outline_head(result.outline)))
    return "\n".join(lines)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("budget must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _notice(structured_data) -> str | None:
    if not isinstance(structured_data, dict):
        return None
    direct = structured_data.get("notice")
    if isinstance(direct, str):
        return direct
    linkedin = structured_data.get("linkedin")
    if isinstance(linkedin, dict) and isinstance(linkedin.get("notice"), str):
        return linkedin["notice"]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
