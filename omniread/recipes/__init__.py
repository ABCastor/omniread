"""yt-dlp-style recipe discovery with no central recipe list."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import pkgutil
from types import ModuleType
from typing import Iterable

from .base import Recipe


class RecipeRegistry:
    """Ordered discovered recipes; specific matches precede the generic fallback."""

    def __init__(self, recipes: Iterable[Recipe]) -> None:
        by_name: dict[str, Recipe] = {}
        for recipe in recipes:
            incumbent = by_name.get(recipe.name)
            if incumbent is None or recipe.priority >= incumbent.priority:
                by_name[recipe.name] = recipe
        self.recipes = tuple(
            sorted(by_name.values(), key=lambda item: item.priority, reverse=True)
        )

    @classmethod
    def discover(
        cls,
        *,
        local_paths: Iterable[Path] | None = None,
        include_entry_points: bool = True,
    ) -> "RecipeRegistry":
        """Discover built-ins, local single-file recipes, and installed entry points."""

        found: list[Recipe] = []
        package = importlib.import_module(__name__)
        for module_info in pkgutil.iter_modules(package.__path__, f"{__name__}."):
            short_name = module_info.name.rsplit(".", 1)[-1]
            if short_name == "base":
                continue
            found.extend(_recipes_in(importlib.import_module(module_info.name)))

        paths = tuple(local_paths) if local_paths is not None else (
            Path.home() / ".omniread/recipes",
        )
        for path in paths:
            if not path.is_dir():
                continue
            for file_path in sorted(path.glob("*.py")):
                if file_path.name.startswith("_"):
                    continue
                module = _load_file(file_path)
                found.extend(_recipes_in(module))

        if include_entry_points:
            for entry_point in importlib.metadata.entry_points(
                group="omniread.recipes"
            ):
                loaded = entry_point.load()
                candidates = loaded if isinstance(loaded, (list, tuple)) else (loaded,)
                for candidate in candidates:
                    found.append(_instantiate(candidate))
        return cls(found)

    def find(self, url: str) -> Recipe:
        """Return the highest-priority matching recipe."""

        for recipe in self.recipes:
            if recipe.match(url):
                return recipe
        raise LookupError("Recipe discovery did not produce a generic fallback")


def _recipes_in(module: ModuleType) -> list[Recipe]:
    recipes: list[Recipe] = []
    for _, candidate in inspect.getmembers(module, inspect.isclass):
        if (
            candidate is not Recipe
            and issubclass(candidate, Recipe)
            and candidate.__module__ == module.__name__
        ):
            recipes.append(candidate())
    return recipes


def _load_file(path: Path) -> ModuleType:
    module_name = f"omniread_local_recipe_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load recipe file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _instantiate(candidate: object) -> Recipe:
    if isinstance(candidate, Recipe):
        return candidate
    if inspect.isclass(candidate) and issubclass(candidate, Recipe):
        return candidate()
    raise TypeError(f"Recipe entry point returned unsupported object {candidate!r}")


__all__ = ["Recipe", "RecipeRegistry"]
