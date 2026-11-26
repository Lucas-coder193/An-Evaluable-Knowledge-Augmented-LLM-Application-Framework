"""Split Markdown documents into metadata-preserving, section-aware JSONL chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text near paragraph/sentence boundaries without dropping characters."""
    if size < 1:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than chunk size")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("。", start, end),
                text.rfind(".", start, end),
            )
            if boundary > start + size // 2:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text.lower()).strip("-")
    return slug or "section"


def markdown_sections(text: str, document_title: str) -> list[tuple[str, str, str]]:
    """Return (section_id, section_title, body); content before the first heading is kept."""
    matches = list(HEADING.finditer(text))
    if not matches:
        body = text.strip()
        return [(slugify(document_title), document_title, body)] if body else []

    sections: list[tuple[str, str, str]] = []
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections.append((slugify(document_title), document_title, preamble))

    seen: dict[str, int] = {}
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        section_id = slugify(title)
        seen[section_id] = seen.get(section_id, 0) + 1
        if seen[section_id] > 1:
            section_id = f"{section_id}-{seen[section_id]}"
        if body:
            sections.append((section_id, title, body))
    return sections


def load_manifest(input_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = input_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["path"]: entry for entry in data.get("documents", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root directory containing Markdown files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    manifest = load_manifest(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in sorted(args.input.rglob("*.md")):
            relative_path = path.relative_to(args.input).as_posix()
            text = path.read_text(encoding="utf-8")
            metadata = manifest.get(relative_path, {})
            title = metadata.get("title", path.stem)
            doc_id = metadata.get("doc_id", slugify(path.stem))
            for section_id, section_title, section_text in markdown_sections(text, title):
                evidence_id = f"{doc_id}#{section_id}"
                for index, chunk in enumerate(split_text(section_text, args.chunk_size, args.overlap)):
                    raw_id = f"{args.project}|{args.version}|{doc_id}|{evidence_id}|{index}"
                    chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
                    row = {
                        "chunk_id": chunk_id,
                        "project": args.project,
                        "version": args.version,
                        "doc_id": doc_id,
                        "evidence_id": evidence_id,
                        "source_path": relative_path,
                        "source_url": metadata.get("source_url", ""),
                        "source_title": title,
                        "section_title": section_title,
                        "chunk_index": index,
                        "text": chunk,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    row_count += 1
    print(f"Wrote {row_count} chunks to {args.output}")


if __name__ == "__main__":
    main()
