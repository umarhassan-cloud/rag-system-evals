from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from rag.config import ChunkingConfig
from rag.models import DocumentChunk

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def load_document(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path}")
    if suffix == ".pdf":
        text = load_pdf(path)
    else:
        text = load_text_file(path)
    return path.name, text


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    doc_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # Prefer breaking on paragraph/sentence boundaries when nearby.
        if end < len(text):
            window = text[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if break_at > chunk_size * 0.4:
                end = start + break_at + 1
        piece = text[start:end].strip()
        if piece:
            chunk_id = _stable_id(doc_id, str(index), piece[:64])
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source=source,
                    text=piece,
                    metadata={"chunk_index": index, "char_start": start, "char_end": end},
                )
            )
            index += 1
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
        # Snap forward to a word boundary so overlapped chunks don't start mid-token.
        while start < len(text) and not text[start].isspace():
            start += 1
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def ingest_paths(paths: list[Path], chunking: ChunkingConfig) -> list[DocumentChunk]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    all_chunks: list[DocumentChunk] = []
    for file_path in files:
        source, text = load_document(file_path)
        doc_id = _stable_id(str(file_path.resolve()), source)
        all_chunks.extend(
            chunk_text(
                text,
                doc_id=doc_id,
                source=source,
                chunk_size=chunking.chunk_size,
                chunk_overlap=chunking.chunk_overlap,
            )
        )
    return all_chunks
