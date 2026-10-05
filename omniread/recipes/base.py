"""Public recipe contract and the narrow services available to recipe plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import re
from typing import ClassVar, Pattern

from curl_cffi import requests

from ..auth import ProfileStore
from ..ladder import Clock, Renderer, clock_iso
from ..tokens import build_outline, cost_for_omitted_sections, truncate_to_budget
from ..types import (
    Completeness,
    CostToComplete,
    JsonValue,
    Provenance,
    ReadResult,
    RetrievalError,
)


@dataclass(frozen=True, slots=True)
class ReadOptions:
    """Caller choices shared by every recipe."""

    budget_tokens: int | None = None
    full: bool = False


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Small transport-neutral response used by structured-data recipes."""

    body: bytes
    status: int
    final_url: str
    headers: dict[str, str]

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> JsonValue:
        return json.loads(self.body)


class RecipeHttpClient:
    """curl_cffi transport shared by recipes; extraction stays in adopted engines."""

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        allow_redirects: bool = True,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        """Fetch one response, optionally without redirects and with a hard byte cap."""

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=45,
                allow_redirects=allow_redirects,
                impersonate="chrome",
                stream=max_bytes is not None,
            )
            status = int(response.status_code)
            redirect = not allow_redirects and 300 <= status < 400
            if not 200 <= status < 300 and not redirect:
                raise RetrievalError(
                    f"Recipe retrieval returned HTTP {status} for {response.url}"
                )
            if max_bytes is None:
                body = bytes(response.content)
            else:
                declared = response.headers.get("content-length")
                if declared is not None and int(declared) > max_bytes:
                    raise RetrievalError(
                        f"Recipe response exceeds the {max_bytes}-byte limit for {url}"
                    )
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    received += len(chunk)
                    if received > max_bytes:
                        raise RetrievalError(
                            f"Recipe response exceeds the {max_bytes}-byte limit for {url}"
                        )
                    chunks.append(bytes(chunk))
                body = b"".join(chunks)
        except requests.errors.RequestsError as exc:
            raise RetrievalError(f"Recipe retrieval failed for {url}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise RetrievalError(f"Recipe retrieval failed for {url}: {exc}") from exc
        finally:
            if "response" in locals() and max_bytes is not None:
                response.close()
        return HttpResponse(
            body=body,
            status=status,
            final_url=str(response.url),
            headers={str(key).lower(): str(value) for key, value in response.headers.items()},
        )


type GenericRead = Callable[..., ReadResult]


@dataclass(frozen=True, slots=True)
class RecipeContext:
    """The intentionally small capability surface passed to recipe code."""

    clock: Clock
    generic_read: GenericRead
    http: RecipeHttpClient
    profiles: ProfileStore
    render: Renderer | None = None
    last_resort: object | None = None


class Recipe:
    """Base class for a single-file site recipe.

    Subclasses declare URL regexes and implement ``read``. Discovery scans for
    subclasses, so adding a recipe never requires a central registry edit.
    """

    name: ClassVar[str]
    URL_PATTERNS: ClassVar[tuple[Pattern[str], ...]] = ()
    priority: ClassVar[int] = 0
    auth_required: ClassVar[bool] = False
    auth_scope: ClassVar[str | None] = None
    RECIPE_META: ClassVar[dict[str, object]] = {}

    @classmethod
    def match(cls, url: str) -> bool:
        """Return whether this recipe owns a URL without doing network work."""

        return any(pattern.search(url) for pattern in cls.URL_PATTERNS)

    @classmethod
    def requires_auth(cls, url: str) -> bool:
        """Return whether this URL must skip anonymous tiers and use Tier 4."""

        return cls.auth_required

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        """Read one URL into the shared result contract."""

        raise NotImplementedError


def shape_recipe_result(
    *,
    url: str,
    markdown: str,
    completeness: Completeness,
    provenance: Provenance,
    structured_data: JsonValue,
    opts: ReadOptions,
    has_pagination: bool = False,
    cost_to_complete: CostToComplete | None = None,
) -> ReadResult:
    """Apply the generic outline and coverage contract to recipe output.

    ``cost_to_complete`` exposes a known retrieval gap on a full-coverage,
    untruncated result. Deliberate budget truncation still takes precedence and
    computes its own omission cost.
    """

    outline = build_outline(markdown, has_pagination=has_pagination)
    bounded = truncate_to_budget(markdown, None if opts.full else opts.budget_tokens)
    return ReadResult(
        url=url,
        content=bounded.content,
        outline=outline,
        completeness=completeness,
        provenance=provenance,
        structured_data=structured_data,
        coverage="core" if bounded.truncated else "full",
        cost_to_complete=(
            cost_for_omitted_sections(outline, bounded.omitted)
            if bounded.truncated
            else cost_to_complete
        ),
        truncated=bounded.truncated,
        omitted=bounded.omitted,
    )


def regexes(*patterns: str) -> tuple[Pattern[str], ...]:
    """Compile recipe URL patterns with the common case-insensitive policy."""

    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
