from __future__ import annotations

from omniread.extract import extract_page
from omniread.probe import probe_structured_data


HTML = """
<html><head>
  <link rel="alternate" type="application/rss+xml" href="/feed.xml">
  <link rel="alternate" type="application/json+oembed" href="/oembed?format=json">
  <script type="application/ld+json">
    {"@type": "Article", "headline": "Companion probes are evidence"}
  </script>
</head><body><main><h1>Companion probes</h1><p>Body text.</p></main></body></html>
"""


def test_tier_zero_probe_collects_jsonld_rss_and_oembed_without_short_circuiting() -> None:
    probe = probe_structured_data(HTML, base_url="https://example.test/article")

    assert probe.json_ld == {
        "@type": "Article",
        "headline": "Companion probes are evidence",
    }
    assert probe.feed_urls == ("https://example.test/feed.xml",)
    assert probe.oembed_urls == ("https://example.test/oembed?format=json",)

    extracted = extract_page(HTML, url="https://example.test/article")
    assert extracted.markdown
    assert extracted.structured_data == probe.as_json()

