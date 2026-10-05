"""Facebook Marketplace item recipe: read the listing from logged-out SSR JSON.

A logged-out ``facebook.com/marketplace/item/<id>`` fetch returns HTTP 200 whose
visible body is a JavaScript shell Trafilatura cannot read, but whose embedded
``<script type="application/json">`` payload already carries the full listing:
title, price, condition, description, and location. So this recipe needs no login
and never touches the browser tier. It parses those JSON blocks, merges the
fragments Facebook splits one listing across (keyed by the item id in the URL),
and returns an honest completeness verdict: ``complete`` only when title, price,
condition, and description were all captured; ``incomplete`` when the listing node
was found but a core field is missing; ``unknown`` when no listing node could be
located at all (Facebook changed its markup, or the item is gated or removed).
It never reports a partial capture as ``complete``.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from selectolax.parser import HTMLParser

from .base import (
    ReadOptions,
    Recipe,
    RecipeContext,
    clock_iso,
    regexes,
    shape_recipe_result,
)
from ..types import Completeness, Evidence, JsonValue, Provenance, ReadResult

_ITEM_ID_RE = re.compile(
    r"^https?://(?:[a-z0-9-]+\.)?facebook\.com/marketplace/item/(\d+)",
    re.IGNORECASE,
)

# Facebook splits a single listing across several GraphQL fragment nodes that all
# share the listing id; these are the fields worth merging out of them.
_LISTING_FIELDS = (
    "marketplace_listing_title",
    "base_marketplace_listing_title",
    "custom_title",
    "listing_price",
    "formatted_price",
    "condition",
    "redacted_description",
    "location_text",
    "location",
    "creation_time",
    "is_live",
    "is_sold",
    "is_pending",
)

_CONDITION_LABELS = {
    "NEW_ITEM": "New",
    "NEW": "New",
    "USED_LIKE_NEW": "Used - Like New",
    "PC_USED_LIKE_NEW": "Used - Like New",
    "USED_GOOD": "Used - Good",
    "PC_USED_GOOD": "Used - Good",
    "USED_FAIR": "Used - Fair",
    "PC_USED_FAIR": "Used - Fair",
    "USED": "Used",
}


class FacebookMarketplaceRecipe(Recipe):
    """Read a Facebook Marketplace item page from its logged-out SSR payload."""

    name = "facebook_marketplace"
    priority = 120
    URL_PATTERNS = regexes(
        r"^https?://(?:[a-z0-9-]+\.)?facebook\.com/marketplace/item/\d+"
    )
    RECIPE_META = {
        "name": name,
        "domains": ["facebook.com"],
        "maturity": "experimental",
        "license_note": "Reads Facebook's own logged-out SSR JSON; selectolax is MIT",
        "test_urls": ["https://www.facebook.com/marketplace/item/1811454709826798"],
    }

    @staticmethod
    def item_id(url: str) -> str:
        """Extract the numeric listing id from the URL without network access."""

        match = _ITEM_ID_RE.match(url)
        if not match:
            raise ValueError(f"Not a recognizable Facebook Marketplace item URL: {url}")
        return match.group(1)

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        item_id = self.item_id(url)
        response = ctx.http.get(url)
        fields = extract_listing(response.text, item_id)
        completeness = verify_listing(fields)
        markdown = build_markdown(fields)
        structured: JsonValue = _structured_data(item_id, fields)
        provenance = Provenance(
            tier=0,
            engine="facebook-marketplace-ssr",
            recipe=self.name,
            canonical_url=f"https://www.facebook.com/marketplace/item/{item_id}",
            fetched_at=clock_iso(ctx.clock),
            http_status=response.status,
            final_url=response.final_url,
            source_urls=[response.final_url],
        )
        return shape_recipe_result(
            url=url,
            markdown=markdown,
            completeness=completeness,
            provenance=provenance,
            structured_data=structured,
            opts=opts,
        )


def _json_blocks(html: str) -> Iterator[JsonValue]:
    """Yield the parsed ``application/json`` script blocks that mention a listing."""

    for node in HTMLParser(html).css('script[type="application/json"]'):
        text = node.text()
        if not text or "marketplace_listing_title" not in text:
            continue
        try:
            yield json.loads(text)
        except (ValueError, TypeError):
            continue


def _walk(value: JsonValue) -> Iterator[dict]:
    """Yield every dict node in a nested JSON value."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def extract_listing(html: str, item_id: str) -> dict:
    """Merge the listing's fields out of every fragment that carries its id.

    Facebook serves one listing as several fragment nodes (a render fragment with
    title/price/condition, a fuller fragment with description/location). Each shares
    ``id == item_id``, so merging first-non-empty per field across all id-matching
    nodes reconstructs the whole listing while ignoring the recommendation carousel
    (whose items carry different ids). Returns an empty dict when no id-matching
    node exists, which the verifier reads as ``unknown``.
    """

    merged: dict = {}
    for block in _json_blocks(html):
        for node in _walk(block):
            if str(node.get("id", "")) != item_id:
                continue
            for field in _LISTING_FIELDS:
                candidate = node.get(field)
                if field not in merged and candidate not in (None, "", {}, []):
                    merged[field] = candidate
    return merged


def _title(fields: dict) -> str:
    for key in ("marketplace_listing_title", "base_marketplace_listing_title", "custom_title"):
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _price(fields: dict) -> str:
    formatted = fields.get("formatted_price")
    if isinstance(formatted, dict):
        text = formatted.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    listing_price = fields.get("listing_price")
    if isinstance(listing_price, dict):
        amount = listing_price.get("amount")
        if isinstance(amount, str) and amount.strip():
            currency = listing_price.get("currency")
            return f"{amount} {currency}".strip() if isinstance(currency, str) else amount
    return ""


def condition_label(code: JsonValue) -> str:
    """Map a Facebook condition enum to a human label, humanizing unknown codes."""

    if not isinstance(code, str) or not code.strip():
        return ""
    known = _CONDITION_LABELS.get(code.upper())
    if known:
        return known
    humanized = re.sub(r"^PC_", "", code.upper()).replace("_", " ").title()
    return humanized


def _description(fields: dict) -> str:
    redacted = fields.get("redacted_description")
    if isinstance(redacted, dict):
        text = redacted.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _location(fields: dict) -> str:
    location_text = fields.get("location_text")
    if isinstance(location_text, dict):
        text = location_text.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def build_markdown(fields: dict) -> str:
    """Render the merged listing into the shared Markdown body."""

    title = _title(fields)
    price = _price(fields)
    condition = condition_label(fields.get("condition"))
    location = _location(fields)
    description = _description(fields)

    facts = []
    if price:
        facts.append(f"- **Price:** {price}")
    if condition:
        facts.append(f"- **Condition:** {condition}")
    if location:
        facts.append(f"- **Location:** {location}")
    if fields.get("is_sold") is True:
        facts.append("- **Status:** Sold")
    elif fields.get("is_pending") is True:
        facts.append("- **Status:** Sale pending")

    parts = [f"# {title}" if title else "# Facebook Marketplace listing"]
    if facts:
        parts.append("\n".join(facts))
    if description:
        parts.append(f"## Description\n\n{description}")
    return "\n\n".join(parts)


def verify_listing(fields: dict) -> Completeness:
    """Judge capture completeness from independent field-presence evidence.

    ``complete`` requires all four core fields (title, price, condition,
    description). A located-but-partial listing is ``incomplete``; a listing whose
    server-rendered node could not be found at all is ``unknown`` and must never be
    reported as ``complete``.
    """

    has_title = bool(_title(fields))
    has_price = bool(_price(fields))
    has_condition = bool(condition_label(fields.get("condition")))
    has_description = bool(_description(fields))

    identity = Evidence(
        name="marketplace_listing_identity",
        passed=has_title,
        detail=(
            "The SSR payload carried the listing title"
            if has_title
            else "No listing title was present in the server-rendered payload"
        ),
    )
    attributes = Evidence(
        name="marketplace_listing_attributes",
        passed=has_price and has_condition,
        detail=_attributes_detail(has_price, has_condition),
    )
    description = Evidence(
        name="marketplace_listing_description",
        passed=has_description,
        detail=(
            "The listing description text was captured"
            if has_description
            else "No listing description text was present in the payload"
        ),
    )
    evidence = [identity, attributes, description]

    if not (has_title or has_price or has_description):
        return Completeness(
            "unknown",
            evidence,
            "Could not locate the listing's server-rendered data; Facebook may have "
            "changed its markup or gated this item.",
        )
    if has_title and has_price and has_condition and has_description:
        return Completeness(
            "complete",
            evidence,
            "Title, price, condition, and description were all captured from "
            "Facebook's own server-rendered payload.",
        )
    return Completeness(
        "incomplete",
        evidence,
        "The listing was located but a core field is missing: "
        + _missing_fields(has_title, has_price, has_condition, has_description),
    )


def _attributes_detail(has_price: bool, has_condition: bool) -> str:
    if has_price and has_condition:
        return "Both the listing price and condition were captured"
    missing = [
        label
        for label, present in (("price", has_price), ("condition", has_condition))
        if not present
    ]
    return f"Missing listing {', '.join(missing)}"


def _missing_fields(
    has_title: bool, has_price: bool, has_condition: bool, has_description: bool
) -> str:
    missing = [
        label
        for label, present in (
            ("title", has_title),
            ("price", has_price),
            ("condition", has_condition),
            ("description", has_description),
        )
        if not present
    ]
    return ", ".join(missing)


def _structured_data(item_id: str, fields: dict) -> JsonValue:
    listing_price = fields.get("listing_price")
    amount = (
        listing_price.get("amount") if isinstance(listing_price, dict) else None
    )
    return {
        "item_id": item_id,
        "title": _title(fields) or None,
        "price_display": _price(fields) or None,
        "price_amount": amount,
        "condition_code": fields.get("condition"),
        "condition": condition_label(fields.get("condition")) or None,
        "location": _location(fields) or None,
        "is_live": fields.get("is_live"),
        "is_sold": fields.get("is_sold"),
        "is_pending": fields.get("is_pending"),
    }
