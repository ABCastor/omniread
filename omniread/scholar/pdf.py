"""Load the optional PDF engine only when PDF bytes are inspected."""

from __future__ import annotations

from io import BytesIO

from ..types import DependencyError


def pdf_reader(payload: bytes):
    """Return a pypdf reader or a useful optional-dependency failure."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DependencyError("PDF reading requires pip install 'omniread[pdf]'") from exc
    return PdfReader(BytesIO(payload), strict=False)
