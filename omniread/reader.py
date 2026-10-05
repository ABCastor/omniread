"""Recipe-routed library core shared by the CLI and MCP peer adapters."""

from __future__ import annotations

import os

import importlib

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .auth import ProfileStore
from . import agent_browser
from .fetch import FetchedPage
from .ladder import Clock, Renderer, clock_iso, read as generic_read
from .recipes import RecipeRegistry
from .recipes.base import ReadOptions, Recipe, RecipeContext, RecipeHttpClient
from .renderer import DefuddleRenderer, render_page
from .tokens import build_outline
from .types import Completeness, Evidence, PolicyError, Provenance, ReadResult

type AuthRendererFactory = Callable[[Path], Renderer]


class Reader:
    """Route a URL through discovered recipes and the generic fallback recipe."""

    def __init__(
        self,
        *,
        registry: RecipeRegistry | None = None,
        http: RecipeHttpClient | None = None,
        profiles: ProfileStore | None = None,
        render: Renderer | None = None,
        auth_renderer_factory: AuthRendererFactory | None = None,
    ) -> None:
        self.registry = registry or RecipeRegistry.discover()
        self.http = http or RecipeHttpClient()
        self.profiles = profiles or ProfileStore()
        self.render = render
        self._custom_auth_renderer = auth_renderer_factory
        self.auth_renderer_factory = auth_renderer_factory or (
            lambda profile: DefuddleRenderer(user_data_dir=profile).render
        )

    def read(
        self,
        url: str,
        *,
        budget: int | None = None,
        full: bool = False,
        clock: Clock | None = None,
        render: Renderer | None = None,
        as_me: bool = False,
    ) -> ReadResult:
        """Read through the matching recipe; the generic ladder is also a recipe."""

        if clock is None:
            raise ValueError("Reader.read() requires a caller-supplied clock")
        recipe = self.registry.find(url)
        browser_available: bool | None = None

        def use_browser() -> bool:
            nonlocal browser_available
            if browser_available is None:
                browser_available = agent_browser.available()
            return browser_available

        required = as_me or recipe.requires_auth(url)
        profiles = replace(self.profiles, browser_managed=required and use_browser())
        if required and not (profiles.browser_managed or self.profiles.profile_dir(url).is_dir()):
            return _auth_required_result(
                url=url, recipe=recipe, profiles=self.profiles, clock=clock,
            )
        anonymous_render = render if render is not None else self.render
        if anonymous_render is None:
            anonymous_render = render_page

        def scoped_generic_read(target_url: str, **kwargs) -> ReadResult:
            target_required = bool(
                kwargs.pop(
                    "auth_required",
                    recipe.match(target_url) and recipe.requires_auth(target_url),
                )
            )
            has_profile = self.profiles.profile_dir(target_url).is_dir()
            profile = self.profiles.profile_dir(target_url) if has_profile else None
            wants_auth = target_required or self.profiles.exists(target_url)

            def authenticated_render(requested_url: str) -> FetchedPage:
                if profile is not None and self._custom_auth_renderer is not None:
                    return self._custom_auth_renderer(profile)(requested_url)
                if use_browser():
                    return agent_browser.GaddiRenderer()(requested_url)
                if profile is not None:
                    return self.auth_renderer_factory(profile)(requested_url)
                raise PolicyError(f"Run omniread login {self.profiles.domain(target_url)} first")

            chosen_render = kwargs.pop("render", anonymous_render)
            kwargs.pop("auth_profile", None)
            kwargs.pop("auth_render", None)
            kwargs.pop("login_command", None)
            domain = self.profiles.domain(target_url)
            return generic_read(
                target_url,
                **kwargs,
                render=chosen_render,
                auth_profile=profile,
                auth_render=authenticated_render if wants_auth else None,
                browser_auth=wants_auth,
                auth_required=target_required,
                login_command=f"omniread login {domain}",
            )

        if as_me:
            return scoped_generic_read(
                url, budget=budget, full=full, clock=clock, auth_required=True,
            )

        context = RecipeContext(
            clock=clock,
            generic_read=scoped_generic_read,
            http=self.http,
            profiles=profiles,
            render=anonymous_render,
            last_resort=_load_last_resort_acquirer(),
        )
        return recipe.read(
            url,
            ReadOptions(budget_tokens=budget, full=full),
            context,
        )


def _load_last_resort_acquirer() -> object | None:
    """Optionally load a locally-installed last-resort acquirer.

    The published core ships NO implementation of this hook and never reaches a
    paywalled paper by any route other than open access or the reader's own
    institutional login (``omniread login``). Anything beyond that is a local
    choice made by the operator on their own machine, not behaviour distributed
    with this package.

    Set ``OMNIREAD_ACQUIRER_PLUGIN`` to a dotted path (``package.module:object``)
    exposing ``acquire(doi, *, resolve) -> outcome``. Absent or unimportable, the
    hook stays ``None`` and the ladder ends at its honest open-access failure.
    """

    spec = os.environ.get("OMNIREAD_ACQUIRER_PLUGIN", "").strip()
    if not spec:
        return None
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    target = getattr(module, attribute, None) if attribute else module
    return target() if isinstance(target, type) else target


_DEFAULT_READER: Reader | None = None


def read(
    url: str,
    *,
    budget: int | None = None,
    full: bool = False,
    clock: Clock | None = None,
    render: Renderer | None = None,
    as_me: bool = False,
) -> ReadResult:
    """Read using the process-local default recipe registry."""

    global _DEFAULT_READER
    if _DEFAULT_READER is None:
        _DEFAULT_READER = Reader()
    return _DEFAULT_READER.read(
        url,
        budget=budget,
        full=full,
        clock=clock,
        render=render,
        as_me=as_me,
    )


def _auth_required_result(
    *,
    url: str,
    recipe: Recipe,
    profiles: ProfileStore,
    clock: Clock,
) -> ReadResult:
    """Fail closed before fetching a site that explicitly requires authentication."""

    domain = profiles.domain(url)
    command = f"omniread login {domain}"
    opt_in = recipe.RECIPE_META.get("opt_in_env")
    reason = f"Authentication is required. Run {command}, then retry."
    if isinstance(opt_in, str) and opt_in:
        reason += f" This default-off recipe also requires {opt_in}=1."
    notice = recipe.RECIPE_META.get("notice")
    structured: dict[str, object] = {
        "auth_required": True,
        "auth_scope": domain,
        "how": command,
    }
    if isinstance(notice, str) and notice:
        structured["notice"] = notice
    return ReadResult(
        url=url,
        content="",
        outline=build_outline(""),
        completeness=Completeness(
            "unknown",
            [
                Evidence(
                    "persistent_profile",
                    False,
                    f"No Chromium profile exists for {domain}",
                )
            ],
            reason,
        ),
        provenance=Provenance(
            tier=None,
            engine="persistent-profile-required",
            recipe=recipe.name,
            canonical_url=url,
            fetched_at=clock_iso(clock),
            http_status=None,
            final_url=url,
            source_urls=[url],
        ),
        structured_data=structured,
        coverage="full",
        cost_to_complete=None,
        truncated=False,
        omitted=[],
    )
