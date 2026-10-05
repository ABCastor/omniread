"""The generic escalation ladder expressed as the fallback recipe."""

from __future__ import annotations

from .base import ReadOptions, Recipe, RecipeContext
from ..types import ReadResult


class GenericRecipe(Recipe):
    """Catch-all recipe. Specific recipes always sort ahead of it."""

    name = "generic"
    priority = -10_000
    RECIPE_META = {
        "name": name,
        "domains": ["*"],
        "maturity": "stable",
        "license_note": "Uses only OmniRead's Apache/MIT engine stack",
        "test_urls": [],
    }

    @classmethod
    def match(cls, url: str) -> bool:
        return True

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        return ctx.generic_read(
            url,
            budget=opts.budget_tokens,
            full=opts.full,
            clock=ctx.clock,
            render=ctx.render,
        )
