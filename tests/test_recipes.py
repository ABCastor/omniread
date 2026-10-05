from __future__ import annotations

from pathlib import Path

from omniread.recipes import RecipeRegistry
from omniread.recipes.academic import AcademicRecipe
from omniread.recipes.linkedin import LinkedInRecipe, classify_linkedin_url
from omniread.recipes.reddit import RedditRecipe
from omniread.recipes.sec_filings import SecFilingsRecipe, _verify_filing
from omniread.recipes.youtube import YouTubeRecipe
from omniread.scholar.extract import extract_jats
from omniread.scholar.locate import AcquiredPaper
from omniread.scholar.metadata import ScholarMetadata
from omniread.scholar.verify import verify_scholarly


def test_builtin_recipes_are_discovered_without_a_central_registry() -> None:
    registry = RecipeRegistry.discover(include_entry_points=False, local_paths=())
    names = {recipe.name for recipe in registry.recipes}

    assert {"academic", "generic", "linkedin", "reddit", "sec_filings", "youtube"} <= names
    assert registry.find("https://www.sec.gov/Archives/edgar/data/1/2/report.htm").name == "sec_filings"
    assert registry.find("https://example.test/article").name == "generic"


def test_single_file_local_recipe_is_auto_discovered(tmp_path: Path) -> None:
    plugin = tmp_path / "private_docs.py"
    plugin.write_text(
        """
import re
from omniread.recipes.base import Recipe

class PrivateDocsRecipe(Recipe):
    name = "private_docs"
    priority = 50
    URL_PATTERNS = (re.compile(r"https://private\\.example/"),)

    def read(self, url, opts, ctx):
        raise NotImplementedError
""",
        encoding="utf-8",
    )

    registry = RecipeRegistry.discover(
        include_entry_points=False,
        local_paths=(tmp_path,),
    )

    assert registry.find("https://private.example/guide").name == "private_docs"


def test_build_now_recipe_matchers_are_narrow_and_url_only() -> None:
    assert SecFilingsRecipe.match(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )
    assert YouTubeRecipe.match("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert AcademicRecipe.match("https://doi.org/10.1371/journal.pmed.0020124")
    assert not AcademicRecipe.match("https://example.test/article")
    assert RedditRecipe.match(
        "https://www.reddit.com/r/Python/comments/abc123/example_thread/"
    )
    assert LinkedInRecipe.match("https://www.linkedin.com/in/example/")


def test_linkedin_classifies_content_type_before_auth_policy() -> None:
    assert classify_linkedin_url("https://www.linkedin.com/company/openai/") == "company"
    assert classify_linkedin_url("https://www.linkedin.com/pulse/example-title/") == "pulse"
    assert (
        classify_linkedin_url(
            "https://www.linkedin.com/posts/example_activity-1234567890"
        )
        == "post"
    )
    profile = "https://www.linkedin.com/in/example/"
    assert classify_linkedin_url(profile) == "profile"
    assert (
        classify_linkedin_url("https://www.linkedin.com/in/example/details/experience/")
        == "profile"
    )
    assert LinkedInRecipe.requires_auth(profile) is True
    assert LinkedInRecipe.requires_auth("https://www.linkedin.com/company/openai/") is False


def test_recipe_input_parsers_preserve_verified_spike_rules() -> None:
    assert YouTubeRecipe.video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert AcademicRecipe.parse_target("https://arxiv.org/abs/1706.03762") == (
        "arxiv",
        "1706.03762",
    )


def test_sec_recipe_requires_both_edgar_manifest_signals() -> None:
    items = ("1", "1A", "2", "3", "4", "5", "7", "7A", "8", "9", "9A", "10", "11", "12", "13", "14")
    markdown = "\n".join(f"## Item {item}.\n\nBody" for item in items)

    complete = _verify_filing(
        form="10-K",
        downloaded_bytes=1200,
        manifest_bytes=1200,
        markdown=markdown,
    )
    missing = _verify_filing(
        form="10-K",
        downloaded_bytes=1200,
        manifest_bytes=1200,
        markdown=markdown.replace("## Item 14.\n\nBody", ""),
    )

    assert complete.status == "complete"
    assert missing.status == "incomplete"


def test_academic_jats_recipe_cross_checks_crossref_reference_count() -> None:
    body = " ".join(f"evidence{i}" for i in range(350))
    payload = f"""
    <article><front><article-meta><title-group><article-title>Verified paper</article-title>
    </title-group><abstract><p>Abstract text.</p></abstract></article-meta></front>
    <body><sec><title>Methods</title><p>{body}</p></sec>
    <sec><title>Results</title><p>{body}</p></sec></body>
    <back><ref-list><ref>First reference</ref><ref>Second reference</ref></ref-list></back>
    </article>
    """.encode()

    extraction = extract_jats(payload)
    paper = AcquiredPaper(
        kind="jats",
        payload=payload,
        rung=1,
        provider="test-jats",
        final_url="https://europepmc.org/articles/PMC1",
        http_status=200,
        source_urls=("https://europepmc.org/articles/PMC1",),
    )
    verdict = verify_scholarly(
        extraction=extraction,
        metadata=ScholarMetadata(
            crossref_reference_count=2,
            openalex_reference_count=2,
        ),
        acquisition=paper,
    )

    assert extraction.reference_count == 2
    assert verdict.completeness.status == "complete"
