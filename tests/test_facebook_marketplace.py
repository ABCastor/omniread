from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from omniread.auth import ProfileStore
from omniread.recipes import RecipeRegistry
from omniread.recipes.base import HttpResponse, ReadOptions, RecipeContext
from omniread.recipes.facebook_marketplace import (
    FacebookMarketplaceRecipe,
    build_markdown,
    condition_label,
    extract_listing,
    verify_listing,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXED_TIME = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)

WHEELS_URL = "https://www.facebook.com/marketplace/item/1000000000000001"
BAG_URL = "https://www.facebook.com/marketplace/item/1000000000000002"


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class _StubHttp:
    """Return fixture bytes for a recipe read without touching the network."""

    def __init__(self, name: str, *, final_url: str) -> None:
        self._body = _html(name).encode("utf-8")
        self._final_url = final_url

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        return HttpResponse(
            body=self._body,
            status=200,
            final_url=self._final_url,
            headers={"content-type": "text/html"},
        )


def _context(tmp_path: Path, http: _StubHttp) -> RecipeContext:
    return RecipeContext(
        clock=lambda: FIXED_TIME,
        generic_read=lambda *args, **kwargs: None,
        http=http,
        profiles=ProfileStore(tmp_path / "profiles"),
        render=None,
    )


def test_matcher_owns_only_item_pages() -> None:
    assert FacebookMarketplaceRecipe.match(WHEELS_URL)
    assert FacebookMarketplaceRecipe.match("https://m.facebook.com/marketplace/item/123")
    assert not FacebookMarketplaceRecipe.match("https://www.facebook.com/marketplace/")
    assert not FacebookMarketplaceRecipe.match("https://www.facebook.com/groups/cycling/")
    assert not FacebookMarketplaceRecipe.match("https://example.test/marketplace/item/1")


def test_item_id_parses_from_the_url() -> None:
    assert FacebookMarketplaceRecipe.item_id(WHEELS_URL) == "1000000000000001"
    assert (
        FacebookMarketplaceRecipe.item_id(BAG_URL + "/?ref=share") == "1000000000000002"
    )


def test_condition_enum_maps_to_human_labels() -> None:
    assert condition_label("PC_USED_GOOD") == "Used - Good"
    assert condition_label("NEW_ITEM") == "New"
    assert condition_label("PC_USED_FAIR") == "Used - Fair"
    assert condition_label("BRAND_SPANKING_NEW") == "Brand Spanking New"
    assert condition_label(None) == ""


def test_wheels_listing_extracts_and_verifies_complete() -> None:
    fields = extract_listing(_html("facebook_marketplace_wheels.html"), "1000000000000001")

    # The recommendation carousel (Volkswagen/BMW, different ids) must be ignored.
    assert fields["marketplace_listing_title"] == "Giant P-R2 700c Road Wheelset - QR Rim Brake"
    assert fields["formatted_price"] == {"text": "£95"}
    assert fields["condition"] == "PC_USED_GOOD"
    assert fields["redacted_description"]["text"].startswith("Giant P-R2 700c road wheelset")

    verdict = verify_listing(fields)
    assert verdict.status == "complete"

    markdown = build_markdown(fields)
    assert "# Giant P-R2 700c Road Wheelset - QR Rim Brake" in markdown
    assert "**Price:** £95" in markdown
    assert "**Condition:** Used - Good" in markdown
    assert "**Location:** Cambridge, Cambridgeshire" in markdown
    assert "quick-release, rim brake" in markdown


def test_bag_listing_extracts_and_verifies_complete() -> None:
    fields = extract_listing(_html("facebook_marketplace_bag.html"), "1000000000000002")

    assert fields["marketplace_listing_title"] == "Decathlon ADVT 900 Bikepacking Bag Set - New"
    assert verify_listing(fields).status == "complete"

    markdown = build_markdown(fields)
    assert "**Price:** £120" in markdown
    assert "**Condition:** New" in markdown
    assert "watertight handlebar dry bag" in markdown


def test_partial_listing_is_incomplete_not_false_complete() -> None:
    fields = extract_listing(
        _html("facebook_marketplace_stripped.html"), "1700000000000000"
    )
    assert fields["marketplace_listing_title"] == "Vintage Camera"

    verdict = verify_listing(fields)
    assert verdict.status == "incomplete"
    assert "price" in verdict.reason and "description" in verdict.reason


def test_gated_page_with_only_carousel_is_unknown() -> None:
    # The target id has no node; only unrelated carousel items are present.
    fields = extract_listing(
        _html("facebook_marketplace_gated.html"), "1000000000000001"
    )
    assert fields == {}
    assert verify_listing(fields).status == "unknown"


def test_read_end_to_end_returns_complete_result(tmp_path: Path) -> None:
    recipe = FacebookMarketplaceRecipe()
    http = _StubHttp("facebook_marketplace_wheels.html", final_url=WHEELS_URL)
    result = recipe.read(WHEELS_URL, ReadOptions(), _context(tmp_path, http))

    assert result.completeness.status == "complete"
    assert result.provenance.tier == 0
    assert result.provenance.recipe == "facebook_marketplace"
    assert result.coverage == "full"
    assert result.structured_data["price_display"] == "£95"
    assert result.structured_data["condition"] == "Used - Good"
    assert result.structured_data["item_id"] == "1000000000000001"
    assert "Giant P-R2 700c Road Wheelset" in result.content


def test_recipe_is_discovered_and_routes_item_urls() -> None:
    registry = RecipeRegistry.discover(include_entry_points=False, local_paths=())
    names = {recipe.name for recipe in registry.recipes}
    assert "facebook_marketplace" in names
    assert registry.find(WHEELS_URL).name == "facebook_marketplace"
    # A non-item Facebook URL must fall through to the generic recipe.
    assert registry.find("https://www.facebook.com/groups/cycling/").name == "generic"
