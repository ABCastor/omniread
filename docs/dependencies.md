# Dependency licenses

OmniRead's own Python code is Apache-2.0. Dependencies are installed separately and remain under their upstream terms. This table describes the direct engines, not a claim that every transitive library has the same license.

| Capability | Engine | Upstream license |
|---|---|---|
| Core text extraction | [Trafilatura](https://github.com/adbar/trafilatura/blob/master/LICENSE) | Apache-2.0 |
| Core URL handling | [Courlan](https://github.com/adbar/courlan/blob/master/LICENSE) | Apache-2.0 |
| Core fetching | [curl_cffi](https://github.com/lexiforest/curl_cffi/blob/main/LICENSE) | MIT; bundled curl and libraries have separate notices |
| Core HTML parsing | [selectolax](https://github.com/rushter/selectolax/blob/master/LICENSE) | MIT; its Lexbor engine is Apache-2.0 |
| Core token counting | [tiktoken](https://github.com/openai/tiktoken/blob/main/LICENSE) | MIT |
| Optional PDF reading (`pdf`, `academic`) | [pypdf](https://github.com/py-pdf/pypdf/blob/main/LICENSE) | BSD-3-Clause |
| Optional OCR (`academic`) | [Docling](https://github.com/docling-project/docling/blob/main/LICENSE) | MIT; model weights and downloaded components have separate terms |
| Optional MCP server (`mcp`) | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE) | MIT |
| Optional video (`youtube`) | [yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/master/LICENSE), [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api/blob/master/LICENSE) | Unlicense and MIT respectively; yt-dlp plugins and distributions may differ |
| Operator-installed rendering | [Playwright](https://github.com/microsoft/playwright/blob/main/LICENSE), Chromium | Apache-2.0; Chromium has BSD and third-party notices |

Transitive Python dependencies include libraries under BSD, PSF and other permissive terms, including lxml and urllib3. Review the installed distributions' license files when redistributing dependencies. No engine, browser, model weight or paid service is bundled by this wheel.

Academic HTML and JATS XML remain available in the core. PDF inspection imports pypdf only on demand; without the extra, PDF extraction fails visibly and the acquisition ladder can continue. No local acquisition-provider implementation is distributed. `OMNIREAD_ACQUIRER_PLUGIN` is an optional operator-supplied `module:object` hook exposing `acquire(doi, *, resolve)`; it is unset by default. An operator remains responsible for that provider's terms and access rights.

The first release bounds Trafilatura to 2.1.x and selectolax to 0.4.x: newer installed releases changed extraction or removed the parser API and failed the existing corpus. MCP support targets SDK 1.x. Bounds should be raised only after the corpus passes with the candidate versions.
