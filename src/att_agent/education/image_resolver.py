"""Convert trusted Gold/Vector Search image metadata into browser image links.

A relative path is not automatically a valid Azure blob key. Set
EDUCATION_IMAGE_BASE_URL only if image_links_json paths are relative to that
public/read-authorized prefix. Private blobs require signed URLs supplied in
metadata or a separately implemented authenticated image service.
"""

from __future__ import annotations

import json
import os
from urllib.parse import quote, urlsplit, urlunsplit


def _absolute_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in ("https", "http") and bool(parsed.netloc)


def _resolve_url(path: str, base_url: str) -> str | None:
    if _absolute_http_url(path):
        return path
    if not base_url or not _absolute_http_url(base_url):
        return None
    parsed = urlsplit(base_url)
    if not path or path.startswith(("/", "\\")):
        return None
    segments = path.replace("\\", "/").split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return None
    encoded_path = "/".join(quote(segment, safe="-_.~") for segment in segments)
    full_path = parsed.path.rstrip("/") + "/" + encoded_path
    return urlunsplit((parsed.scheme, parsed.netloc, full_path, parsed.query, ""))


def resolve_document_images(doc: dict, base_url: str | None = None) -> list[dict]:
    """Normalize image_links_json into images with URL and source_path.

    Supports lists of strings or mappings. No URL is fabricated when a blob
    relative path cannot be resolved. Does not parse arbitrary Markdown images:
    Markdown-relative paths cannot reliably identify an Azure blob.
    """
    raw = doc.get("image_links_json")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    base = base_url if base_url is not None else os.getenv("EDUCATION_IMAGE_BASE_URL", "")
    images: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            path = item
            alt = "Educational illustration"
        elif isinstance(item, dict):
            path = next(
                (
                    item[key]
                    for key in (
                        "url",
                        "image_url",
                        "signed_url",
                        "blob_url",
                        "blob_path",
                        "path",
                        "source_path",
                        "href",
                    )
                    if isinstance(item.get(key), str) and item[key]
                ),
                "",
            )
            alt = str(
                item.get("alt_text")
                or item.get("alt")
                or item.get("caption")
                or "Educational illustration"
            )
        else:
            continue
        path = path.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        images.append(
            {
                "url": _resolve_url(path, base),
                "source_path": path,
                "alt_text": alt,
            }
        )
    return images
