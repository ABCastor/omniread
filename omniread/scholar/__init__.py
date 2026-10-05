"""Scholarly artifact acquisition and content-level honesty."""

from .ids import ScholarlyIdentifier, is_scholarly_input, normalize_identifier
from .verify import ContentLevel, ScholarVerdict

__all__ = [
    "ContentLevel",
    "ScholarVerdict",
    "ScholarlyIdentifier",
    "is_scholarly_input",
    "normalize_identifier",
]
