"""Portability boundaries that must hold in a core-only installation."""
import subprocess
import sys
from pathlib import Path

from omniread import renderer


def test_core_imports_and_extracts_without_optional_pdf_engine():
    code = """
import sys
sys.modules['pypdf'] = None
from omniread import reader
from omniread.extract_api import extract_html
from omniread.scholar.pdf import pdf_reader
from omniread.types import DependencyError
response = extract_html('<html><body><p>Short supplied content.</p></body></html>', 'https://example.test')
assert response['result']['content']
try:
    pdf_reader(b'%PDF-1.4')
except DependencyError as exc:
    assert '[pdf]' in str(exc)
else:
    raise AssertionError('Missing PDF dependency was silently accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_browser_root_is_operator_configurable(monkeypatch):
    monkeypatch.setenv("OMNIREAD_PLAYWRIGHT_ROOT", "~/browser-bundle")
    assert renderer._find_defuddle_bundle() == Path.home() / "browser-bundle"
    monkeypatch.delenv("OMNIREAD_PLAYWRIGHT_ROOT")
    monkeypatch.setenv("OMNIREAD_DEFUDDLE_PLAYWRIGHT_ROOT", "~/legacy-bundle")
    assert renderer._find_defuddle_bundle() == Path.home() / "legacy-bundle"
    monkeypatch.delenv("OMNIREAD_DEFUDDLE_PLAYWRIGHT_ROOT")
    assert renderer._find_defuddle_bundle() == Path.home() / ".local/share/omniread/browser"


def test_uncertain_page_extent_cannot_be_overruled_by_intact_references():
    from omniread.scholar.extract import ScholarExtraction
    from omniread.scholar.locate import AcquiredPaper
    from omniread.scholar.metadata import ScholarMetadata
    from omniread.scholar.verify import verify_scholarly

    metadata = ScholarMetadata(
        first_page="1", last_page="10",
        crossref_reference_count=100, openalex_reference_count=100,
    )
    paper = AcquiredPaper(
        "pdf", b"%PDF", 4, "repository", "https://example.test/paper.pdf",
        200, ("https://example.test/paper.pdf",),
    )
    for body_words, expected in [(1_800, "unknown"), (3_600, "complete")]:
        markdown = (
            "# Paper\n\n## Methods\n\n" + "observation " * (body_words // 2)
            + "\n\n## Results\n\n" + "finding " * (body_words // 2)
            + "\n\n## References\n\nReference list."
        )
        extraction = ScholarExtraction(
            markdown, ("Methods", "Results", "References"), 100, True,
            len(markdown.split()),
        )
        verdict = verify_scholarly(extraction=extraction, metadata=metadata, acquisition=paper)
        evidence = {item.name: item.passed for item in verdict.completeness.evidence}
        assert evidence["section_manifest"] is True
        assert evidence["reference_manifest"] is True
        assert verdict.completeness.status == expected
        if expected == "unknown":
            assert evidence["page_extent"] is None


def test_unlabelled_next_links_are_content_gaps_and_other_documents_are_not():
    from omniread.extract import extract_page
    from omniread.fetch import FetchedPage, detect_block_page
    from omniread.verify import verify_completeness

    url = "https://example.test/story"
    base = "<html><body><article><h1>Story</h1><h2>One</h2><p>" + "alpha "*200 + "</p><h2>Two</h2><p>" + "beta "*200 + "</p>"
    controls = [
        ('<a href="/story/2">Next</a>', "incomplete"),
        ('<a href="?page=2">Next</a>', "incomplete"),
        ('<a href="?page=2" aria-label="Next page">→</a>', "incomplete"),
        ('<a href="?page=2" aria-label="Next page">Continue</a>', "incomplete"),
        ('<a href="?page=2">→</a>', "incomplete"),
        ('<a href="?page=2">Next →</a>', "incomplete"),
        ('<nav><a href="?page=2">2</a></nav>', "incomplete"),
        ('<a href="/other-story">Next</a>', "complete"),
        ('<a href="/other-story" aria-label="Next page">→</a>', "complete"),
        ('<nav><a href="/other-story">2</a></nav>', "complete"),
    ]
    for control, expected in controls:
        html = base + control + '</article></body></html>'
        fetched = FetchedPage(html, 200, url, {}, detect_block_page(html, http_status=200, final_url=url))
        assert verify_completeness(fetched, extract_page(html, url=url)).status == expected, control



def test_reddit_declared_count_gap_cannot_earn_complete():
    from omniread.recipes.reddit import parse_redlib_thread, verify_comment_capture
    for count, expected in [(500, "incomplete"), (1, "complete")]:
        html = f'<html><body><h1>Thread</h1><p>{count} comments</p><div class="comment"><div class="comment_body">A captured comment.</div></div></body></html>'
        thread = parse_redlib_thread(html, reddit_url="https://www.reddit.com/r/test/comments/abc123/title/")
        assert verify_comment_capture(thread).status == expected


def test_optional_pdf_hint_survives_extraction_and_unlock_routing(monkeypatch):
    from omniread.scholar.extract import extract_pdf
    from omniread.scholar.locate import AcquisitionResult
    from omniread.scholar.recipe import _unlock_path
    from omniread.types import DependencyError
    import pytest
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(DependencyError) as failure:
        extract_pdf(b"%PDF-1.4")
    assert "pdf extra" in _unlock_path(acquisition=AcquisitionResult(), extraction_error=str(failure.value))


def test_renderer_adapter_runs_with_browser_default_user_agent(tmp_path):
    import json
    import shutil
    import pytest
    if not shutil.which("node"):
        pytest.skip("Node.js is optional")
    (tmp_path / "package.json").write_text("{}")
    module = tmp_path / "node_modules/playwright"
    module.mkdir(parents=True)
    (module / "index.js").write_text("""
const page = {
  on() {}, mainFrame() { return this; }, url() { return 'https://example.test'; },
  async goto() { return { status: () => 200 }; }, async waitForTimeout() {},
  async content() { return '<html><body>Rendered adapter fixture.</body></html>'; }
};
const context = { pages: () => [page] };
module.exports = { chromium: { async launch() { return {
  async newContext(options) {
    if ('userAgent' in options) throw new Error('Unexpected fixed user agent');
    return context;
  }, async close() {}
}; } } };
""")
    script = Path(renderer.__file__).with_name("browser_render.mjs")
    result = subprocess.run(["node", str(script), "https://example.test", "--playwright-root", str(tmp_path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == 200


def test_public_result_urls_do_not_echo_query_credentials():
    from omniread.extract_api import extract_html
    import json
    response = extract_html('<html><body><p>Supplied representation.</p></body></html>', 'https://example.test/article?token=private-value#account')
    assert 'private-value' not in json.dumps(response)
    assert response['result']['url'] == 'https://example.test/article'


def test_heading_word_inside_paragraph_does_not_prove_a_missing_section():
    from dataclasses import replace
    from omniread.extract import extract_page
    from omniread.fetch import FetchedPage, detect_block_page
    from omniread.verify import verify_completeness
    url = 'https://example.test/story'
    html = '<article><h1>Story</h1><h2>Introduction</h2><p>' + ('These results illustrate the observed method. '*80) + '</p><h2>Results</h2><p>A distinct small section.</p></article>'
    fetched = FetchedPage(html, 200, url, {}, detect_block_page(html, http_status=200, final_url=url))
    extracted = extract_page(html, url=url)
    assert verify_completeness(fetched, extracted).status == 'complete'
    missing = replace(extracted, markdown=extracted.markdown.split('## Results')[0])
    assert verify_completeness(fetched, missing).status == 'unknown'
