<h1><img src="docs/media/readme-header.svg" width="112" height="72" align="absmiddle" alt=""> <img src="docs/omniread-title.svg" width="175" align="absmiddle" alt="OmniRead*"></h1>

OmniRead is a Python web reader for AI agents that returns Markdown with a completeness verdict: `complete`, `incomplete`, or `unknown`.

[![Checks](https://github.com/ABCastor/omniread/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/ABCastor/omniread/actions/workflows/checks.yml)

An extracted page can contain readable prose while missing entire sections. OmniRead compares the extraction against evidence from the source, including headings, declared word counts and pagination, so callers can see gaps before relying on the text.

## Works with Gaddi

[Gaddi](https://github.com/ABCastor/gaddi) gives an agent access to your existing Chrome tabs and logins. OmniRead turns the captured page into Markdown, a section index and evidence of what may be missing. Gaddi’s `browser_read` already uses OmniRead when it is installed.

That combination matters on a signed-in article: the agent can read the page you can access, ask for a particular section, and see when only an abstract, a blocked page or part of a longer document was captured. Gaddi controls browser actions and approvals; OmniRead checks the reading. Its completeness verdict remains evidence, not a guarantee.

## Install

Requires [Python](https://www.python.org/) 3.13 or later. Install from source in a virtual environment:

```sh
git clone https://github.com/ABCastor/omniread.git
cd omniread
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
omniread read 'https://example.com' --json
```

Optional extras add capabilities:

```sh
python -m pip install '.[pdf]'       # PDF papers
python -m pip install '.[mcp]'       # MCP server
python -m pip install '.[youtube]'   # YouTube transcripts
python -m pip install '.[academic]'  # PDF papers and Docling OCR
```

[Trafilatura](https://github.com/adbar/trafilatura) extracts the text; [curl_cffi](https://github.com/lexiforest/curl_cffi) retrieves pages. Downloaded dependencies keep their own licenses, listed in [dependency notes](docs/dependencies.md).

## Read and extract

```sh
omniread read 'https://example.com/article' --json --budget 2000
omniread extract --html page.html --url 'https://example.com/article' --json
```

`read` retrieves the URL. `extract` uses supplied HTML without network or browser access and returns a `{handle, result}` envelope. Both include content, an outline, completeness evidence and provenance.

The verdict describes the captured source. `complete` requires independent positive signals; `incomplete` identifies a measured gap; `unknown` means the evidence cannot establish either. A block page cannot earn `complete`. A token budget sets `coverage` to `core` and names omitted sections separately from the acquisition verdict.

Recipes cover academic papers, Reddit, YouTube, SEC filings and Facebook Marketplace. Their endpoints and public mirrors can change or refuse access. Add a Python recipe under `~/.omniread/recipes/` to extend the reader.

**Completeness is evidence, not a guarantee.** The tests cover synthetic truncations, block pages and selected site representations. They do not establish an error rate across the web. PDF section and page-count heuristics can miss excerpts, and a rendered page can still hide content. Inspect the evidence before using a result where missing material matters.

## Browser and login support

[Gaddi](https://github.com/ABCastor/gaddi) lets OmniRead read through your existing logged-in Chrome on macOS. Start Gaddi, then use:

```sh
omniread login example.com
omniread read 'https://example.com/article' --as-me --json
```

Authenticated reads open a background tab, capture its HTML and close only that tab. Gaddi holds and denials stay visible; OmniRead cannot approve them. Set `GADDI_SOCKET` if your broker uses a non-default socket. Browser content, including private material visible after login, is returned to the calling agent.

Public reads use HTTP first. For optional JavaScript rendering, install [Playwright](https://playwright.dev/) and its isolated Chromium in a directory you control:

```sh
mkdir -p "$HOME/.local/share/omniread/browser"
cd "$HOME/.local/share/omniread/browser"
npm init -y
npm install playwright
npx playwright install chromium
```

Requires [Node.js](https://nodejs.org/) 22 or later. This default directory is discovered automatically; `OMNIREAD_PLAYWRIGHT_ROOT` selects another directory containing `package.json` and Playwright. The legacy `OMNIREAD_DEFUDDLE_PLAYWRIGHT_ROOT` remains supported. The renderer uses bundled Chromium and never selects your system Chrome. Without Gaddi, `login` can create a separate persistent Chromium profile.

## MCP and development

```sh
omniread-mcp
```

After installing the `mcp` extra, this runs a [Model Context Protocol](https://modelcontextprotocol.io/) stdio server exposing `read_url`, `read_section` and `read_more`.

```sh
python -m pip install '.[dev,pdf,mcp]'
python -m pytest -q
ruff check --select E9,F63,F7,F82 omniread tests scripts
python -m build
```

## License

[Apache-2.0](LICENSE). The title's outlined lettering uses [Literata](docs/OFL-literata.txt), licensed under the SIL Open Font License. External engines and optional providers retain their own terms.

<p><a href="https://abcastor.com"><img src="docs/castor-footer.svg" width="350" alt="Chip, the Castor beaver, by Castor, we give a dam"></a></p>
