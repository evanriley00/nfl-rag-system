from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.services.chunking import Chunk, chunk_text


SUPPORTED_EXTENSIONS = {".md", ".txt"}
UPLOADABLE_EXTENSIONS = {".md", ".txt", ".pdf"}
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class Document:
    document_id: str
    title: str
    path: str
    content: str


def slugify(value: str) -> str:
    normalized = SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized or "document"


def titleize_slug(value: str) -> str:
    return value.replace("-", " ").title()


def load_documents(documents_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        document_id = slugify(path.stem)
        documents.append(
            Document(
                document_id=document_id,
                title=titleize_slug(document_id),
                path=str(path),
                content=content,
            )
        )
    return documents


def build_chunks(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_text(
                document_id=document.document_id,
                title=document.title,
                text=document.content,
            )
        )
    return chunks


def save_manifest(documents: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(document) for document in documents]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_uploaded_document(filename: str, payload: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOADABLE_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload .txt, .md, or .pdf files.")

    if suffix in {".txt", ".md"}:
        try:
            return payload.decode("utf-8").strip(), suffix
        except UnicodeDecodeError as exc:
            raise ValueError("Text uploads must be UTF-8 encoded.") from exc

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            "PDF upload requires pypdf. Run `pip install -r requirements.txt` first."
        ) from exc

    from io import BytesIO

    reader = PdfReader(BytesIO(payload))
    extracted_pages = [(page.extract_text() or "").strip() for page in reader.pages]
    content = "\n\n".join(page for page in extracted_pages if page)
    if not content:
        raise ValueError("No readable text was found in the uploaded PDF.")
    return content, suffix


def save_uploaded_document(*, documents_dir: Path, filename: str, payload: bytes) -> dict[str, str]:
    content, original_suffix = parse_uploaded_document(filename, payload)
    if not content:
        raise ValueError("Uploaded document was empty after parsing.")

    stem = slugify(Path(filename).stem)
    stored_path = documents_dir / f"{stem}.txt"
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_text(content, encoding="utf-8")
    return {
        "document_id": stem,
        "title": titleize_slug(stem),
        "stored_path": str(stored_path),
        "source_type": original_suffix.removeprefix("."),
    }
