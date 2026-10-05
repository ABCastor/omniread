"""Extract scholarly bodies with adopted engines and preserve artifact structure."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

from .pdf import pdf_reader
from selectolax.parser import HTMLParser

from ..extract import extract_page
from ..types import DependencyError, ExtractionError
from .locate import AcquiredPaper
from .xml import parse_jats_xml


@dataclass(frozen=True, slots=True)
class ScholarExtraction:
    """Structured observations made from acquired paper bytes."""

    markdown: str
    section_titles: tuple[str, ...]
    reference_count: int
    references_resolved: bool
    word_count: int


def extract_paper(paper: AcquiredPaper) -> ScholarExtraction:
    """Extract one verified representation into structured Markdown."""

    if paper.kind == "jats":
        return extract_jats(paper.payload)
    if paper.kind == "html":
        return extract_html(paper.payload.decode("utf-8", errors="replace"), paper.final_url)
    if paper.kind == "pdf":
        return extract_pdf(paper.payload)
    return extract_markdown(paper.payload.decode("utf-8", errors="replace"))


def extract_jats(payload: bytes) -> ScholarExtraction:
    """Convert JATS XML while retaining nested sections and the reference list."""

    try:
        root = parse_jats_xml(payload)
    except (ValueError, SyntaxError) as exc:
        raise ExtractionError(f"JATS XML could not be parsed safely: {exc}") from exc
    title = _node_text(_first(root, "article-title")) or "Academic paper"
    parts = [f"# {title}"]
    section_titles: list[str] = []

    abstract = _first(root, "abstract")
    abstract_text = _node_text(abstract)
    if abstract_text:
        parts.append(f"## Abstract\n\n{abstract_text}")
        section_titles.append("Abstract")

    body = _first(root, "body")
    if body is None or not _node_text(body):
        raise ExtractionError("JATS document contains no article body")
    for child in body:
        if _local_name(child.tag) == "sec":
            rendered, titles = _render_jats_section(child, level=2)
            if rendered:
                parts.append(rendered)
                section_titles.extend(titles)
        else:
            block = _render_block(child)
            if block:
                parts.append(block)

    ref_list = _first(root, "ref-list")
    references = (
        [
            node
            for node in ref_list.iter()
            if _local_name(node.tag) == "ref"
        ]
        if ref_list is not None
        else []
    )
    if references:
        rendered_refs = [
            f"{index}. {_node_text(node)}"
            for index, node in enumerate(references, 1)
            if _node_text(node)
        ]
        if rendered_refs:
            parts.append("## References\n\n" + "\n".join(rendered_refs))
            section_titles.append("References")

    markdown = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    return ScholarExtraction(
        markdown=markdown,
        section_titles=tuple(section_titles),
        reference_count=len(references),
        references_resolved=bool(references),
        word_count=len(markdown.split()),
    )


def extract_html(html: str, url: str) -> ScholarExtraction:
    """Use OmniRead's Trafilatura extraction seam for scholarly HTML."""

    markdown = extract_page(html, url=url).markdown.strip()
    if not markdown:
        raise ExtractionError("Scholarly HTML yielded no readable body")
    section_titles = tuple(_markdown_headings(markdown))
    return ScholarExtraction(
        markdown=markdown,
        section_titles=section_titles,
        reference_count=_markdown_reference_count(markdown),
        # HTML bibliography text is observed, but unlike JATS it is not a resolved
        # structured manifest. It therefore does not earn the stronger level.
        references_resolved=False,
        word_count=len(markdown.split()),
    )


def extract_pdf(payload: bytes) -> ScholarExtraction:
    """Extract born-digital PDFs with pypdf; OCR only when no text layer exists."""

    if not payload.startswith(b"%PDF"):
        raise ExtractionError("PDF extraction received a non-PDF representation")
    if has_text_layer(payload):
        return extract_markdown(_pypdf_markdown(payload))
    return extract_markdown(_docling_markdown(payload))


def has_text_layer(payload: bytes, *, sample_pages: int = 3) -> bool:
    """Return whether pypdf observes text in the first pages of a PDF."""

    try:
        reader = pdf_reader(payload)
        return any(
            (page.extract_text() or "").strip()
            for page in reader.pages[:sample_pages]
        )
    except DependencyError:
        raise
    except Exception as exc:
        raise ExtractionError(f"pypdf could not inspect the PDF text layer: {exc}") from exc


def _pypdf_markdown(payload: bytes) -> str:
    try:
        reader = pdf_reader(payload)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise ExtractionError(f"pypdf could not extract the scholarly PDF: {exc}") from exc
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ExtractionError("pypdf returned no scholarly content")
    return _promote_pdf_headings(text)


def _docling_markdown(payload: bytes) -> str:
    """OCR a text-layer-free PDF with optional MIT-licensed Docling."""

    try:
        converter_module = importlib.import_module("docling.document_converter")
    except ImportError as exc:
        raise DependencyError(
            "Scanned scholarly PDF extraction requires optional Docling (MIT)"
        ) from exc
    try:
        with tempfile.TemporaryDirectory(prefix="omniread-scholar-") as directory:
            path = Path(directory) / "paper.pdf"
            path.write_bytes(payload)
            converted = converter_module.DocumentConverter().convert(str(path))
            markdown = str(converted.document.export_to_markdown()).strip()
    except Exception as exc:
        raise ExtractionError(f"Docling could not convert the scholarly PDF: {exc}") from exc
    if not markdown:
        raise ExtractionError("Docling returned no scholarly content")
    return markdown


_PDF_SECTION_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)[.)]?\s+)?"
    r"(?:abstract|introduction|background|related work|methods?|materials and methods|"
    r"results?|discussion|conclusions?|references|bibliography|acknowledg(?:e)?ments?)$",
    re.IGNORECASE,
)


def _promote_pdf_headings(text: str) -> str:
    """Preserve standard paper sections that pypdf emits as plain lines."""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 120 and _PDF_SECTION_HEADING.fullmatch(stripped):
            lines.append(f"## {stripped}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def extract_markdown(markdown: str) -> ScholarExtraction:
    """Observe sections and references in authenticated or PDF Markdown."""

    cleaned = markdown.strip()
    if not cleaned:
        raise ExtractionError("Scholarly Markdown is empty")
    return ScholarExtraction(
        markdown=cleaned,
        section_titles=tuple(_markdown_headings(cleaned)),
        reference_count=_markdown_reference_count(cleaned),
        references_resolved=False,
        word_count=len(cleaned.split()),
    )


def extract_abstract_from_html(html: str) -> str | None:
    """Extract a landing-page abstract without mistaking page chrome for the paper."""

    tree = HTMLParser(html)
    for name in ("citation_abstract", "dc.description", "description"):
        for node in tree.css("meta[name]"):
            if (node.attributes.get("name") or "").strip().lower() != name:
                continue
            content = " ".join((node.attributes.get("content") or "").split())
            if content:
                return content
    for selector in (
        '[data-title="Abstract"]',
        '[data-title="abstract"]',
        "#Abs1-content",
        ".abstract",
        ".c-article-section__content",
    ):
        node = tree.css_first(selector)
        if node:
            text = " ".join(node.text().split())
            if text:
                return text
    return None


def extract_title_from_html(html: str) -> str | None:
    """Return a scholarly title from citation metadata or the first H1."""

    tree = HTMLParser(html)
    for node in tree.css("meta[name]"):
        if (node.attributes.get("name") or "").strip().lower() == "citation_title":
            title = " ".join((node.attributes.get("content") or "").split())
            if title:
                return title
    heading = tree.css_first("h1")
    if heading:
        title = " ".join(heading.text().split())
        if title:
            return title
    return None


def _render_jats_section(
    section: ET.Element,
    *,
    level: int,
) -> tuple[str, list[str]]:
    title_node = next(
        (child for child in section if _local_name(child.tag) == "title"),
        None,
    )
    title = _node_text(title_node) or "Section"
    parts = [f"{'#' * min(level, 6)} {title}"]
    titles = [title]
    for child in section:
        name = _local_name(child.tag)
        if name == "title":
            continue
        if name == "sec":
            rendered, nested_titles = _render_jats_section(child, level=level + 1)
            if rendered:
                parts.append(rendered)
                titles.extend(nested_titles)
            continue
        block = _render_block(child)
        if block:
            parts.append(block)
    return "\n\n".join(parts), titles


def _render_block(node: ET.Element) -> str:
    name = _local_name(node.tag)
    if name in {"p", "disp-quote", "boxed-text", "statement", "caption"}:
        return _node_text(node)
    if name == "list":
        values = [
            _node_text(item)
            for item in node.iter()
            if _local_name(item.tag) == "list-item"
        ]
        return "\n".join(f"- {value}" for value in values if value)
    if name in {"table-wrap", "fig"}:
        label = _node_text(next((child for child in node if _local_name(child.tag) == "label"), None))
        caption = _node_text(next((child for child in node if _local_name(child.tag) == "caption"), None))
        return ": ".join(value for value in (label, caption) if value)
    return ""


def _markdown_headings(markdown: str) -> list[str]:
    return [
        match.group(2).strip()
        for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", markdown, re.MULTILINE)
        if len(match.group(1)) >= 2
    ]


def _markdown_reference_count(markdown: str) -> int:
    reference_heading = re.search(
        r"^#{1,6}\s+(?:references|bibliography)\s*$",
        markdown,
        re.IGNORECASE | re.MULTILINE,
    )
    if reference_heading is None:
        # Heading-less artifacts (accepted-article and scanned PDFs) still carry
        # the numbered list in their tail. Counting the bounded tail keeps the
        # inline body citations, which sit mid-sentence and never match the
        # reference patterns, out of the manifest.
        lines = markdown.splitlines()
        tail = "\n".join(lines[int(len(lines) * 0.6) :])
    else:
        tail = markdown[reference_heading.end() :]
        next_heading = re.search(r"^#{1,6}\s+\S", tail, re.MULTILINE)
        if next_heading:
            tail = tail[: next_heading.start()]
    numbers = [
        int(captured)
        for captured in re.findall(
            r"(?:\[\s*(\d{1,4})\s*\]|^[\s\-*]*(\d{1,4})[.)]\s+\S)",
            tail,
            re.MULTILINE,
        )
        for captured in (captured[0] or captured[1],)
    ]
    # Extracted PDF text merges several entries onto one line, so the highest
    # seen reference number is the most faithful count of the list.
    return max(numbers) if numbers else 0


def _first(root: ET.Element, local_name: str) -> ET.Element | None:
    return next(
        (node for node in root.iter() if _local_name(node.tag) == local_name),
        None,
    )


def _node_text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
