"""Retry transient scholarly HTTP failures without confusing them with absence."""

from __future__ import annotations

from collections.abc import Callable
import re
import time

from ..recipes.base import HttpResponse, RecipeHttpClient
from ..types import RetrievalError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)


def get_with_retry(
    http: RecipeHttpClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    allow_redirects: bool = True,
    max_bytes: int | None = None,
    tries: int = 4,
    sleeper: Callable[[float], None] | None = None,
) -> HttpResponse:
    """GET once successfully or raise after bounded exponential backoff.

    HTTP 429 and transient 5xx responses are availability failures, never evidence
    that a scholarly record or artifact is absent. Non-transient HTTP responses and
    other retrieval errors fail immediately.
    """

    if tries < 1:
        raise ValueError("tries must be at least one")
    pause = sleeper or time.sleep
    delay = 1.0
    last_detail = "transient retrieval failure"
    for attempt in range(1, tries + 1):
        try:
            response = http.get(
                url,
                headers=headers,
                allow_redirects=allow_redirects,
                max_bytes=max_bytes,
            )
        except RetrievalError as exc:
            status = _status_from_error(str(exc))
            if status not in RETRY_STATUSES:
                raise
            last_detail = f"HTTP {status}"
        else:
            if response.status in RETRY_STATUSES:
                last_detail = f"HTTP {response.status}"
            elif 200 <= response.status < 300 or 300 <= response.status < 400:
                return response
            else:
                raise RetrievalError(
                    f"Scholarly retrieval returned HTTP {response.status} for {url}"
                )
        if attempt < tries:
            pause(delay)
            delay *= 2
    raise RetrievalError(
        f"Scholarly retrieval exhausted {tries} attempts after {last_detail} for {url}"
    )


def _status_from_error(detail: str) -> int | None:
    match = _STATUS_RE.search(detail)
    return int(match.group(1)) if match else None

