"""Default-off LinkedIn routing with user-owned Tier-4 profiles only."""

from __future__ import annotations

from dataclasses import replace
import os
import re

from .base import ReadOptions, Recipe, RecipeContext, clock_iso, regexes
from ..tokens import build_outline
from ..types import Completeness, Evidence, Provenance, ReadResult

LINKEDIN_OPT_IN_ENV = "OMNIREAD_LINKEDIN_ENABLED"
LINKEDIN_NOTICE = (
    "LinkedIn prohibits anonymous automated access in robots.txt; this opt-in recipe "
    "uses only the user's own stored login for personal profiles."
)


def classify_linkedin_url(url: str) -> str:
    """Classify the URL before choosing an anonymous or authenticated tier."""

    if re.search(r"/in/[^/?#]+(?:/|[?#]|$)", url, re.I):
        return "profile"
    if re.search(r"/company/[^/?#]+", url, re.I):
        return "company"
    if re.search(r"/pulse/[^/?#]+", url, re.I):
        return "pulse"
    if re.search(r"/posts/[^?#]*[-_]activity-\d+", url, re.I) or "/feed/update/" in url:
        return "post"
    return "other"


class LinkedInRecipe(Recipe):
    """Read opted-in LinkedIn types without ever probing a profile anonymously."""

    name = "linkedin"
    priority = 120
    auth_required = True
    auth_scope = "linkedin.com"
    URL_PATTERNS = regexes(r"^https?://(?:[a-z]{2,3}\.)?linkedin\.com/")
    RECIPE_META = {
        "name": name,
        "domains": ["linkedin.com"],
        "auth_scope": auth_scope,
        "enabled_by_default": False,
        "opt_in_env": LINKEDIN_OPT_IN_ENV,
        "maturity": "experimental",
        "license_note": "No added dependency; public types use the Apache/MIT core stack",
        "notice": LINKEDIN_NOTICE,
        "test_urls": [
            "https://www.linkedin.com/company/linkedin/",
            "https://www.linkedin.com/pulse/example-article/",
            "https://www.linkedin.com/in/example/",
        ],
    }

    @classmethod
    def match(cls, url: str) -> bool:
        return super().match(url) and classify_linkedin_url(url) != "other"

    @classmethod
    def requires_auth(cls, url: str) -> bool:
        return classify_linkedin_url(url) == "profile"

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        kind = classify_linkedin_url(url)
        opted_in = _opted_in()
        has_profile = ctx.profiles.exists(url)
        if not opted_in or not has_profile:
            missing = []
            if not opted_in:
                missing.append(f"set {LINKEDIN_OPT_IN_ENV}=1")
            if not has_profile:
                missing.append(f"run omniread login {ctx.profiles.domain(url)}")
            return _disabled_result(url, kind=kind, missing=missing, ctx=ctx)

        result = ctx.generic_read(
            url,
            budget=opts.budget_tokens,
            full=opts.full,
            clock=ctx.clock,
            render=ctx.render,
        )
        structured = {
            "linkedin": {
                "url_type": kind,
                "notice": LINKEDIN_NOTICE,
            },
            "page": result.structured_data,
        }
        return replace(
            result,
            provenance=replace(result.provenance, recipe=self.name),
            structured_data=structured,
        )


def _opted_in() -> bool:
    return os.environ.get(LINKEDIN_OPT_IN_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _disabled_result(
    url: str,
    *,
    kind: str,
    missing: list[str],
    ctx: RecipeContext,
) -> ReadResult:
    instruction = " and ".join(missing)
    reason = f"The default-off LinkedIn recipe was not activated; {instruction}."
    return ReadResult(
        url=url,
        content="",
        outline=build_outline(""),
        completeness=Completeness(
            "unknown",
            [
                Evidence(
                    "linkedin_opt_in_policy",
                    False,
                    reason,
                )
            ],
            reason,
        ),
        provenance=Provenance(
            tier=None,
            engine="linkedin-default-off",
            recipe="linkedin",
            canonical_url=url,
            fetched_at=clock_iso(ctx.clock),
            http_status=None,
            final_url=url,
            source_urls=[url],
        ),
        structured_data={
            "linkedin": {
                "url_type": kind,
                "notice": LINKEDIN_NOTICE,
                "enabled": False,
            }
        },
        coverage="full",
        cost_to_complete=None,
        truncated=False,
        omitted=[],
    )
