from __future__ import annotations

from dataclasses import fields

import pytest

from omniread.types import Completeness, CostToComplete, Evidence, Provenance


def test_completeness_contract_has_no_percentage_field() -> None:
    assert [field.name for field in fields(Completeness)] == ["status", "evidence", "reason"]


def test_completeness_rejects_status_outside_tri_state() -> None:
    with pytest.raises(ValueError, match="Invalid completeness status"):
        Completeness(status="mostly", evidence=[], reason="invalid")  # type: ignore[arg-type]


def test_provenance_requires_caller_clock_with_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        Provenance(
            tier=1,
            engine="test",
            recipe="generic",
            canonical_url="https://example.test/",
            fetched_at="2026-07-12T12:00:00",
            http_status=200,
            final_url="https://example.test/",
            source_urls=["https://example.test/"],
        )


def test_provenance_strips_query_and_fragment_secrets_from_public_urls() -> None:
    provenance = Provenance(
        tier=4,
        engine="test",
        recipe="generic",
        canonical_url="https://example.test/article?public-filter=latest",
        fetched_at="2026-07-13T12:00:00+00:00",
        http_status=200,
        final_url="https://example.test/article?session=secret#account",
        source_urls=[
            "https://example.test/start?token=secret",
            "https://example.test/article?session=secret#account",
            "https://example.test/article#duplicate-after-sanitizing",
        ],
    )

    assert provenance.canonical_url == "https://example.test/article"
    assert provenance.final_url == "https://example.test/article"
    assert provenance.source_urls == [
        "https://example.test/start",
        "https://example.test/article",
    ]


def test_evidence_supports_unknown_signal() -> None:
    evidence = Evidence(name="manifest", passed=None, detail="No manifest is available")
    assert evidence.passed is None


def test_cost_to_complete_is_concrete_not_a_confidence_score() -> None:
    cost = CostToComplete(remaining_items=3, estimated_extra_tokens=420)

    assert cost.remaining_items == 3
    assert cost.estimated_extra_tokens == 420
    assert cost.required_tier is None
    assert cost.estimated_latency_ms is None


def test_cost_to_complete_rejects_negative_measurements() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CostToComplete(remaining_items=-1, estimated_extra_tokens=0)
