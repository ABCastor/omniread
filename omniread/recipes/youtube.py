"""YouTube transcript and metadata recipe using the adopted yt-dlp libraries."""

from __future__ import annotations

import importlib
import re

from .base import (
    ReadOptions,
    Recipe,
    RecipeContext,
    clock_iso,
    regexes,
    shape_recipe_result,
)
from ..types import (
    Completeness,
    DependencyError,
    Evidence,
    JsonValue,
    Provenance,
    ReadResult,
    RetrievalError,
)


class YouTubeRecipe(Recipe):
    """Read the video itself: metadata, description, chapters, and transcript.

    Comments are explicitly outside this dispatch's recipe scope. The verified
    spike showed that yt-dlp exposes only a top-sorted sample while overwriting the
    true total; treating that sample as full would violate the new coverage axis.
    """

    name = "youtube"
    priority = 100
    URL_PATTERNS = regexes(
        r"^https?://(?:www\.)?youtube\.com/(?:watch\?.*\bv=|shorts/|live/)",
        r"^https?://youtu\.be/",
    )
    RECIPE_META = {
        "name": name,
        "domains": ["youtube.com", "youtu.be"],
        "maturity": "experimental",
        "license_note": "yt-dlp is Unlicense; youtube-transcript-api is MIT",
        "test_urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    }

    @staticmethod
    def video_id(url: str) -> str:
        """Extract the stable eleven-character video id without network access."""

        match = re.search(
            r"(?:[?&]v=|/shorts/|/live/|youtu\.be/)([A-Za-z0-9_-]{11})",
            url,
        )
        if not match:
            raise ValueError(f"Not a recognizable YouTube video URL: {url}")
        return match.group(1)

    def read(self, url: str, opts: ReadOptions, ctx: RecipeContext) -> ReadResult:
        yt_dlp, transcript_api = _libraries()
        video_id = self.video_id(url)
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(
                {"skip_download": True, "quiet": True, "no_warnings": True}
            ) as ydl:
                info = ydl.extract_info(canonical, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise RetrievalError(f"yt-dlp could not retrieve YouTube metadata: {exc}") from exc

        transcript, transcript_evidence = _transcript(transcript_api, video_id)
        metadata_ok = bool(info.get("title"))
        metadata_evidence = Evidence(
            name="youtube_player_metadata",
            passed=metadata_ok,
            detail=(
                "yt-dlp parsed the player response and returned a title"
                if metadata_ok
                else "The player response did not contain a video title"
            ),
        )
        evidence = [metadata_evidence, transcript_evidence]
        if any(item.passed is False for item in evidence):
            completeness = Completeness(
                "incomplete", evidence, "The declared video scope is missing metadata or transcript"
            )
        elif all(item.passed is True for item in evidence):
            completeness = Completeness(
                "complete", evidence, "Independent player metadata and caption track agree"
            )
        else:
            completeness = Completeness(
                "unknown", evidence, "The transcript could not be independently verified"
            )

        description = str(info.get("description") or "").strip()
        chapter_lines = [
            f"- {chapter.get('start_time', 0):.0f}s: {chapter.get('title', 'Untitled')}"
            for chapter in (info.get("chapters") or [])
        ]
        markdown = "\n\n".join(
            part
            for part in (
                f"# {info.get('title') or video_id}",
                f"## Description\n\n{description}" if description else "",
                "## Chapters\n\n" + "\n".join(chapter_lines) if chapter_lines else "",
                f"## Transcript\n\n{transcript}" if transcript else "",
            )
            if part
        )
        structured: JsonValue = {
            "content_scope": "video metadata, description, chapters, and transcript; comments excluded",
            "video_id": video_id,
            "channel": info.get("channel") or info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration_seconds": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "reported_comment_count": info.get("comment_count"),
        }
        provenance = Provenance(
            tier=0,
            engine="yt-dlp+youtube-transcript-api",
            recipe=self.name,
            canonical_url=canonical,
            fetched_at=clock_iso(ctx.clock),
            http_status=None,
            final_url=canonical,
            source_urls=[canonical],
        )
        return shape_recipe_result(
            url=url,
            markdown=markdown,
            completeness=completeness,
            provenance=provenance,
            structured_data=structured,
            opts=opts,
        )


def _libraries():
    missing: list[str] = []
    modules: list[object] = []
    for name in ("yt_dlp", "youtube_transcript_api"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            missing.append(name.replace("_", "-"))
    if missing:
        raise DependencyError(
            "YouTube recipe requires adopted libraries: " + ", ".join(missing)
        )
    return modules[0], modules[1]


def _transcript(api_module, video_id: str) -> tuple[str, Evidence]:
    try:
        listing = api_module.YouTubeTranscriptApi().list(video_id)
        tracks = list(listing)
        if not tracks:
            return "", Evidence(
                "youtube_caption_track", False, "YouTube exposes no transcript track"
            )
        chosen = next((track for track in tracks if not track.is_generated), tracks[0])
        fetched = chosen.fetch()
        snippets = getattr(fetched, "snippets", fetched)
        text = " ".join(
            str(snippet.text).strip()
            for snippet in snippets
            if str(snippet.text).strip()
        )
        return text, Evidence(
            "youtube_caption_track",
            bool(text),
            f"Fetched {len(snippets)} signed caption snippets from language {chosen.language_code}",
        )
    except Exception as exc:
        return "", Evidence(
            "youtube_caption_track",
            None,
            f"Caption retrieval could not be judged: {type(exc).__name__}",
        )
