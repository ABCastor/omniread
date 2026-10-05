"""Bounded XML parsing for third-party scholarly representations."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

MAX_JATS_BYTES = 16 * 1024 * 1024
_ENTITY_DECLARATION = re.compile(br"<!ENTITY\b", re.IGNORECASE)


def parse_jats_xml(payload: bytes) -> ET.Element:
    """Parse JATS without permitting unbounded entity expansion."""

    if len(payload) > MAX_JATS_BYTES:
        raise ValueError(
            f"JATS response exceeds the {MAX_JATS_BYTES}-byte input limit"
        )
    if _ENTITY_DECLARATION.search(payload):
        raise ValueError("JATS response contains forbidden entity declarations")
    return ET.fromstring(payload)
