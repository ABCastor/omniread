"""OmniRead's typed web-reading core."""

from .reader import Reader, read
from .types import (
    Completeness,
    CostToComplete,
    Evidence,
    Outline,
    Provenance,
    ReadResult,
    Section,
)

__all__ = [
    "Completeness",
    "CostToComplete",
    "Evidence",
    "Outline",
    "Provenance",
    "ReadResult",
    "Reader",
    "Section",
    "read",
]

__version__ = "0.1.0"
