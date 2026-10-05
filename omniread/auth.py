"""Per-site login preferences and fallback Chromium profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from courlan import extract_domain

from .types import PolicyError

PROFILE_DIR_ENV = "OMNIREAD_PROFILE_DIR"
DEFAULT_PROFILE_DIR = Path("~/.config/omniread/profiles")


def normalize_site_url(value: str) -> str:
    """Return an absolute HTTP(S) URL from either a URL or a bare site name."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("A URL or site name is required")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise PolicyError("Login profiles require an HTTP(S) URL or site name")
    if parsed.username is not None or parsed.password is not None:
        raise PolicyError("Do not put credentials in a login URL")
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def registrable_domain(value: str) -> str:
    """Resolve a URL or site to its registrable domain using Courlan's PSL data."""

    normalized = normalize_site_url(value)
    host = (urlsplit(normalized).hostname or "").lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        domain = extract_domain(normalized)
        if domain:
            return domain.lower().rstrip(".")
    else:
        return host

    # Courlan deliberately rejects private/synthetic suffixes such as ``.test``.
    # Keeping their full host makes local fixtures and intranet profiles usable
    # without pretending to know a registrable boundary that does not exist.
    if host and all(part and part.replace("-", "").isalnum() for part in host.split(".")):
        return host
    raise PolicyError(f"Cannot determine a safe profile scope for {value!r}")


def profile_root() -> Path:
    """Return the external profile root, honoring ``OMNIREAD_PROFILE_DIR``."""

    configured = os.environ.get(PROFILE_DIR_ENV)
    return Path(configured or DEFAULT_PROFILE_DIR).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ProfileStore:
    """Map sites to login preferences and fallback Chromium profiles."""

    root: Path = field(default_factory=profile_root)
    browser_managed: bool = False

    @property
    def base_dir(self) -> Path:
        return self.root.expanduser().resolve()

    def domain(self, value: str) -> str:
        """Return the profile scope for a URL or bare site."""

        return registrable_domain(value)

    def profile_dir(self, value: str) -> Path:
        """Return, but do not create, the scoped Chromium profile directory."""

        return self.base_dir / self.domain(value)

    def exists(self, value: str) -> bool:
        """Whether a site has a login preference, local profile, or Chrome capability.

        This permits an authenticated attempt; it never certifies a valid login.
        """

        path = self.profile_dir(value)
        return (
            self.browser_managed or path.is_dir()
            or (self.base_dir / f"{self.domain(value)}.chrome-login").is_file()
        )

    def remember_chrome_login(self, value: str) -> None:
        """Record a non-secret site preference, never Chrome cookies or credentials."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / f"{self.domain(value)}.chrome-login").touch()

    def prepare(self, value: str) -> Path:
        """Create only the non-secret parent and return the future profile path."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self.profile_dir(value)


def profile_dir_for(value: str) -> Path:
    """Convenience wrapper around the default profile store."""

    return ProfileStore().profile_dir(value)


def profile_exists(value: str) -> bool:
    """Convenience wrapper around the default profile store."""

    return ProfileStore().exists(value)
