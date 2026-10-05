from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from omniread import ladder
from omniread.auth import ProfileStore, normalize_site_url, registrable_domain
from omniread.fetch import FetchedPage, detect_block_page
from omniread.reader import Reader
from omniread.recipes import RecipeRegistry
from omniread.recipes.base import Recipe
from omniread.recipes.generic import GenericRecipe
from omniread.recipes.linkedin import LINKEDIN_NOTICE, LinkedInRecipe

FIXTURES = Path(__file__).parent / "fixtures"
FIXED_TIME = datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)


def _page(name: str, *, url: str) -> FetchedPage:
    raw_html = (FIXTURES / name).read_text(encoding="utf-8")
    return FetchedPage(
        raw_html=raw_html,
        http_status=200,
        final_url=url,
        headers={"content-type": "text/html"},
        block_signal=detect_block_page(raw_html, http_status=200, final_url=url),
    )


def test_profile_store_uses_external_env_root_and_registrable_domain(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "external-profiles"
    monkeypatch.setenv("OMNIREAD_PROFILE_DIR", str(root))
    store = ProfileStore()

    assert store.root == root
    assert normalize_site_url("nytimes.com") == "https://nytimes.com/"
    assert registrable_domain("https://news.bbc.co.uk/story") == "bbc.co.uk"
    assert store.profile_dir("https://www.nytimes.com/article") == root / "nytimes.com"
    assert store.exists("https://www.nytimes.com/article") is False
    store.profile_dir("nytimes.com").mkdir(parents=True)
    assert store.exists("https://cooking.nytimes.com/recipe") is True


def test_authwall_without_profile_fails_closed_with_login_instruction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    url = "https://subscriber.example.com/article"
    monkeypatch.setattr(
        ladder,
        "fetch_static",
        lambda target: _page("authwall_page.html", url=target),
    )
    reader = Reader(
        registry=RecipeRegistry((GenericRecipe(),)),
        profiles=ProfileStore(tmp_path / "profiles"),
        auth_renderer_factory=lambda profile: (_ for _ in ()).throw(
            AssertionError("Tier 4 must not run without a profile")
        ),
    )

    result = reader.read(url, clock=lambda: FIXED_TIME)

    assert result.completeness.status == "unknown"
    assert "Run omniread login example.com" in result.completeness.reason
    assert result.content == ""
    assert result.provenance.tier == 1


def test_authwall_with_profile_escalates_to_tier4_and_marks_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    url = "https://subscriber.example.com/article"
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir(url).mkdir(parents=True)
    monkeypatch.setattr(
        ladder,
        "fetch_static",
        lambda target: _page("authwall_page.html", url=target),
    )
    observed: list[Path] = []

    def factory(profile: Path):
        observed.append(profile)
        return lambda target: _page("normal_article.html", url=target)

    result = Reader(
        registry=RecipeRegistry((GenericRecipe(),)),
        profiles=profiles,
        auth_renderer_factory=factory,
    ).read(url, clock=lambda: FIXED_TIME)

    assert observed == [profiles.profile_dir(url)]
    assert result.completeness.status == "complete"
    assert result.provenance.tier == 4
    assert "persistent-profile" in result.provenance.engine


class _AuthRequiredRecipe(Recipe):
    name = "auth_fixture"
    priority = 10
    auth_required = True

    @classmethod
    def match(cls, url: str) -> bool:
        return "auth-required.example.com" in url

    def read(self, url, opts, ctx):
        return ctx.generic_read(
            url,
            budget=opts.budget_tokens,
            full=opts.full,
            clock=ctx.clock,
        )


def test_declared_auth_required_skips_anonymous_fetch_and_needs_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    url = "https://auth-required.example.com/private"
    profiles = ProfileStore(tmp_path / "profiles")
    static_calls: list[str] = []
    monkeypatch.setattr(ladder, "fetch_static", lambda target: static_calls.append(target))
    reader = Reader(
        registry=RecipeRegistry((_AuthRequiredRecipe(), GenericRecipe())),
        profiles=profiles,
    )

    missing = reader.read(url, clock=lambda: FIXED_TIME)
    assert missing.completeness.status == "unknown"
    assert "omniread login example.com" in missing.completeness.reason
    assert static_calls == []

    profiles.profile_dir(url).mkdir(parents=True)
    reader = Reader(
        registry=RecipeRegistry((_AuthRequiredRecipe(), GenericRecipe())),
        profiles=profiles,
        auth_renderer_factory=lambda profile: (
            lambda target: _page("normal_article.html", url=target)
        ),
    )
    authenticated = reader.read(url, clock=lambda: FIXED_TIME)
    assert authenticated.provenance.tier == 4
    assert static_calls == []


def test_linkedin_is_default_off_even_with_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    url = "https://www.linkedin.com/in/example/"
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir(url).mkdir(parents=True)
    monkeypatch.delenv("OMNIREAD_LINKEDIN_ENABLED", raising=False)
    monkeypatch.setattr(
        ladder,
        "fetch_static",
        lambda target: (_ for _ in ()).throw(
            AssertionError("default-off LinkedIn must not fetch")
        ),
    )
    result = Reader(
        registry=RecipeRegistry((LinkedInRecipe(), GenericRecipe())),
        profiles=profiles,
    ).read(url, clock=lambda: FIXED_TIME)

    assert result.completeness.status == "unknown"
    assert "OMNIREAD_LINKEDIN_ENABLED=1" in result.completeness.reason
    assert result.structured_data["linkedin"]["notice"] == LINKEDIN_NOTICE


def test_opted_in_linkedin_routes_open_types_static_and_profiles_tier4(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.profile_dir("linkedin.com").mkdir(parents=True)
    monkeypatch.setenv("OMNIREAD_LINKEDIN_ENABLED", "1")
    static_calls: list[str] = []

    def static(target: str) -> FetchedPage:
        static_calls.append(target)
        return _page("normal_article.html", url=target)

    auth_calls: list[str] = []

    def auth_factory(profile: Path):
        def authenticated(target: str) -> FetchedPage:
            auth_calls.append(target)
            return _page("normal_article.html", url=target)

        return authenticated

    monkeypatch.setattr(ladder, "fetch_static", static)
    reader = Reader(
        registry=RecipeRegistry((LinkedInRecipe(), GenericRecipe())),
        profiles=profiles,
        auth_renderer_factory=auth_factory,
    )
    company = reader.read(
        "https://www.linkedin.com/company/openai/",
        clock=lambda: FIXED_TIME,
    )
    profile = reader.read(
        "https://www.linkedin.com/in/example/",
        clock=lambda: FIXED_TIME,
    )

    assert company.provenance.tier == 1
    assert company.provenance.recipe == "linkedin"
    assert static_calls == ["https://www.linkedin.com/company/openai/"]
    assert profile.provenance.tier == 4
    assert profile.provenance.recipe == "linkedin"
    assert auth_calls == ["https://www.linkedin.com/in/example/"]
