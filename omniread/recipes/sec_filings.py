"""SEC EDGAR filing recipe adapted from the verified dispatch spikes."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit, urlunsplit

from .base import (
    ReadOptions,
    Recipe,
    RecipeContext,
    clock_iso,
    regexes,
    shape_recipe_result,
)
from ..extract import extract_page
from ..types import Completeness, Evidence, JsonValue, PolicyError, Provenance, ReadResult

_CORE_ITEMS = {
    "10-K": ("1", "1A", "2", "3", "4", "5", "7", "7A", "8", "9", "9A", "10", "11", "12", "13", "14"),
    "10-Q": ("1", "2", "3", "4"),
}


class SecFilingsRecipe(Recipe):
    """Read an EDGAR primary filing and verify it against EDGAR's byte manifest."""

    name = "sec_filings"
    priority = 100
    URL_PATTERNS = regexes(
        r"^https?://(?:www\.)?sec\.gov/Archives/edgar/data/\d+/\d+/[^/?#]+\.html?(?:[?#]|$)"
    )
    RECIPE_META = {
        "name": name,
        "domains": ["sec.gov"],
        "maturity": "stable",
        "license_note": "Public US government EDGAR API; Trafilatura is Apache-2.0",
        "test_urls": [
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
        ],
    }

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        user_agent = os.environ.get("OMNIREAD_SEC_USER_AGENT", "").strip()
        if not user_agent:
            raise PolicyError(
                "SEC requires a descriptive User-Agent with real contact details; set "
                "OMNIREAD_SEC_USER_AGENT='Name contact@example.com'"
            )
        headers = {"user-agent": user_agent, "accept-encoding": "identity"}
        document = ctx.http.get(url, headers=headers)
        index_url = _index_url(document.final_url)
        index = ctx.http.get(index_url, headers=headers)
        manifest_size = _manifest_size(index.json(), _document_name(document.final_url))

        raw_html = document.text
        cleaned = re.sub(r"<ix:header.*?</ix:header>", "", raw_html, flags=re.I | re.S)
        cleaned = re.sub(r"<ix:hidden.*?</ix:hidden>", "", cleaned, flags=re.I | re.S)
        extracted = extract_page(cleaned, url=document.final_url)
        form = _form_type(raw_html)
        completeness = _verify_filing(
            form=form,
            downloaded_bytes=len(document.body),
            manifest_bytes=manifest_size,
            markdown=extracted.markdown,
        )
        structured: JsonValue = {
            "probe": extracted.structured_data,
            "sec": {
                "form": form,
                "downloaded_bytes": len(document.body),
                "manifest_bytes": manifest_size,
            },
        }
        provenance = Provenance(
            tier=0,
            engine="sec-edgar-json+trafilatura",
            recipe=self.name,
            canonical_url=extracted.canonical_url,
            fetched_at=clock_iso(ctx.clock),
            http_status=document.status,
            final_url=document.final_url,
            source_urls=[document.final_url, index.final_url],
        )
        return shape_recipe_result(
            url=url,
            markdown=extracted.markdown,
            completeness=completeness,
            provenance=provenance,
            structured_data=structured,
            opts=opts,
        )


def _index_url(url: str) -> str:
    parts = urlsplit(url)
    parent = parts.path.rsplit("/", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, f"{parent}/index.json", "", ""))


def _document_name(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1]


def _manifest_size(value: JsonValue, document_name: str) -> int | None:
    if not isinstance(value, dict):
        return None
    directory = value.get("directory")
    if not isinstance(directory, dict):
        return None
    items = directory.get("item")
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("name") == document_name:
            size = item.get("size")
            if isinstance(size, (int, str)) and str(size).isdigit():
                return int(size)
    return None


def _form_type(raw_html: str) -> str | None:
    for pattern in (
        r"<TYPE>\s*(10-K|10-Q)\b",
        r"\bFORM\s+(10-K|10-Q)\b",
    ):
        if match := re.search(pattern, raw_html, re.IGNORECASE):
            return match.group(1).upper()
    return None


def _verify_filing(
    *,
    form: str | None,
    downloaded_bytes: int,
    manifest_bytes: int | None,
    markdown: str,
) -> Completeness:
    byte_match = manifest_bytes is not None and downloaded_bytes == manifest_bytes
    byte_evidence = Evidence(
        name="edgar_byte_manifest",
        passed=byte_match if manifest_bytes is not None else None,
        detail=(
            f"Downloaded {downloaded_bytes} bytes; EDGAR declares {manifest_bytes}"
            if manifest_bytes is not None
            else "EDGAR's index did not expose a byte manifest for the primary document"
        ),
    )
    core = _CORE_ITEMS.get(form or "")
    if core is None:
        item_evidence = Evidence(
            name="statutory_item_manifest",
            passed=None,
            detail="The filing form could not be identified as 10-K or 10-Q",
        )
    else:
        found = {
            match.upper()
            for match in re.findall(
                r"^[ \t]*(?:#+\s*)?Item\s+([0-9]+[A-Za-z]?)\.?",
                markdown,
                flags=re.I | re.M,
            )
        }
        missing = [item for item in core if item not in found]
        item_evidence = Evidence(
            name="statutory_item_manifest",
            passed=not missing,
            detail=(
                f"Found all {len(core)} core {form} Items"
                if not missing
                else f"Missing core {form} Items: {', '.join(missing)}"
            ),
        )
    evidence = [byte_evidence, item_evidence]
    if byte_evidence.passed is False or item_evidence.passed is False:
        return Completeness("incomplete", evidence, "EDGAR's independent manifest identifies a gap")
    if all(item.passed is True for item in evidence):
        return Completeness("complete", evidence, "EDGAR byte and statutory Item manifests agree")
    return Completeness(
        "unknown",
        evidence,
        "The filing lacks enough independent manifest evidence to certify extraction",
    )
