"""Create an auditable snapshot record for a curated Markdown corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from .chunk_documents import markdown_sections, split_text
else:
    from chunk_documents import markdown_sections, split_text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_snapshot(
    corpus_root: Path,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("manifest must declare a non-empty documents list")
    declared = {item.get("path"): item for item in documents if isinstance(item, dict)}
    if len(declared) != len(documents) or None in declared:
        raise ValueError("manifest document paths must be unique non-empty strings")
    markdown_paths = {path.relative_to(corpus_root).as_posix() for path in corpus_root.rglob("*.md")}
    if markdown_paths != set(declared):
        missing = sorted(set(declared) - markdown_paths)
        unlisted = sorted(markdown_paths - set(declared))
        raise ValueError(f"manifest/files mismatch; missing={missing}, unlisted={unlisted}")
    doc_ids = [str(item.get("doc_id", "")) for item in documents]
    if any(not value for value in doc_ids) or len(doc_ids) != len(set(doc_ids)):
        raise ValueError("manifest doc_id values must be non-empty and unique")

    total_chunks = 0
    frozen_documents = []
    for relative_path in sorted(declared):
        metadata = declared[relative_path]
        source_url = metadata.get("source_url", "")
        if not isinstance(source_url, str) or not source_url.startswith("https://fastapi.tiangolo.com/"):
            raise ValueError(f"document source must be an official FastAPI tutorial URL: {relative_path}")
        path = corpus_root / relative_path
        content = path.read_text(encoding="utf-8")
        chunk_count = sum(
            len(split_text(section_text, chunk_size, overlap))
            for _, _, section_text in markdown_sections(content, metadata.get("title", path.stem))
        )
        total_chunks += chunk_count
        frozen_documents.append({
            "path": relative_path,
            "doc_id": metadata["doc_id"],
            "title": metadata.get("title", path.stem),
            "source_url": source_url,
            "sha256": sha256(path),
            "characters": len(content),
            "chunks": chunk_count,
        })

    return {
        "dataset_id": manifest.get("dataset_id"),
        "description": manifest.get("description"),
        "snapshot_date": manifest.get("snapshot_date"),
        "upstream_project": manifest.get("upstream_project"),
        "upstream_repository": manifest.get("upstream_repository"),
        "upstream_license": manifest.get("upstream_license"),
        "upstream_license_url": manifest.get("upstream_license_url"),
        "manifest_sha256": sha256(manifest_path),
        "chunking": {"algorithm": "heading-aware-overlap-v1", "chunk_size": chunk_size, "overlap": overlap},
        "documents_count": len(frozen_documents),
        "chunks_count": total_chunks,
        "documents": frozen_documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/fastapi/frozen-v2"))
    parser.add_argument("--output", type=Path, default=Path("data/snapshots/fastapi_frozen_v2.json"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()
    snapshot = build_snapshot(args.corpus, chunk_size=args.chunk_size, overlap=args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"frozen {snapshot['documents_count']} docs / {snapshot['chunks_count']} chunks -> {args.output}")


if __name__ == "__main__":
    main()
