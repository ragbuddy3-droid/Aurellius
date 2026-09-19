import json
import os
import re
from pathlib import Path

from adaptive_chunking import build_chunks, label_blocks, split_blocks_on_sections
from extract_pdf import extract_blocks


DATA_DIR = Path("data")
PJOK_EXTRA_DIR = DATA_DIR / "pjok_tambahan"
OTHER_RESEARCH_DIR = DATA_DIR / "penelitian_lain"
OUT_DIR = Path("out")
UPLOAD_BOOKS_DIR = Path("uploads") / "teacher_books"
UPLOAD_METADATA_DIR = UPLOAD_BOOKS_DIR / "metadata"

OUT_DIR.mkdir(exist_ok=True)
PJOK_EXTRA_DIR.mkdir(parents=True, exist_ok=True)
OTHER_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def slugify_code(value: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", " ", value or "")
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    cleaned = cleaned.strip("._ ")
    return cleaned.upper() or "BOOK"


def get_uploaded_book_codes() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not UPLOAD_METADATA_DIR.exists():
        return mapping

    for metadata_path in sorted(UPLOAD_METADATA_DIR.glob("*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        saved_pdf = payload.get("saved_pdf")
        if not saved_pdf:
            continue

        subject = payload.get("subject") or "BOOK"
        grade = payload.get("grade") or ""
        title = payload.get("book_title") or Path(saved_pdf).stem
        code_parts = [slugify_code(subject)]
        if grade:
            code_parts.append(f"KELAS_{slugify_code(str(grade))}")
        code_parts.append(slugify_code(title))
        mapping[saved_pdf] = "_".join(part for part in code_parts if part)

    return mapping


def collect_pdf_files() -> list[Path]:
    files: list[Path] = []

    if DATA_DIR.exists():
        files.extend(
            sorted(
                path
                for path in DATA_DIR.rglob("*.pdf")
                if path.is_file()
            )
        )

    if UPLOAD_BOOKS_DIR.exists():
        files.extend(
            sorted(
                path
                for path in UPLOAD_BOOKS_DIR.rglob("*.pdf")
                if path.is_file()
                and path.parent != UPLOAD_METADATA_DIR
            )
        )

    return files


PDF_FILES = collect_pdf_files()
UPLOADED_BOOK_CODES = get_uploaded_book_codes()


def get_book_code(pdf_path: str) -> str:
    filename = os.path.basename(pdf_path)
    if filename in UPLOADED_BOOK_CODES:
        return UPLOADED_BOOK_CODES[filename]
    base = os.path.splitext(filename)[0]
    return slugify_code(base)


def save_txt(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['chunk_id']}\n")
            f.write(f"SOURCE_BOOK: {r['source_book']}\n")
            f.write(f"CHAPTER: {r.get('chapter')}\n")
            f.write(f"CHAPTER_NUMBER: {r.get('chapter_number')}\n")
            f.write(f"HEADINGS: {r['headings']}\n")
            f.write(f"CATEGORIES: {r['categories']}\n")
            f.write(f"CUE_PATTERNS: {r['cue_patterns']}\n")
            f.write("CONTENT:\n")
            f.write(r["content"])
            f.write("\n" + "-" * 80 + "\n")


all_chunks = []
total_chunks = 0

for pdf_path in PDF_FILES:
    if not pdf_path.exists():
        print(f"File tidak ditemukan: {pdf_path}")
        continue

    book_code = get_book_code(str(pdf_path))
    print(f"\n=== Processing {book_code} ===")

    blocks = extract_blocks(str(pdf_path))
    blocks = split_blocks_on_sections(blocks)
    labeled_blocks = label_blocks(blocks)
    chunks = build_chunks(labeled_blocks)

    book_rows = []

    for ch in chunks:
        row = {
            "chunk_id": f"{book_code}_{ch.chunk_id}",
            "source_book": book_code,
            "source_pdf": str(pdf_path),
            "chapter": ch.meta.get("chapter"),
            "chapter_number": ch.meta.get("chapter_number"),
            "headings": ch.meta.get("headings", []),
            "block_ids": ch.block_ids,
            "cue_patterns": ch.cue_patterns,
            "categories": ch.categories,
            "content": ch.content,
            "meta": ch.meta,
        }
        book_rows.append(row)
        all_chunks.append(row)

    json_path = OUT_DIR / f"chunks_{book_code}.jsonl"
    with open(json_path, "w", encoding="utf-8") as f:
        for r in book_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    txt_path = OUT_DIR / f"chunks_{book_code}.txt"
    save_txt(txt_path, book_rows)

    print(f"Saved {len(book_rows)} chunks for {book_code}")
    total_chunks += len(book_rows)


json_all_path = OUT_DIR / "chunks_all.jsonl"
with open(json_all_path, "w", encoding="utf-8") as f:
    for r in all_chunks:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

txt_all_path = OUT_DIR / "chunks_all.txt"
save_txt(txt_all_path, all_chunks)

print(f"\nTotal {total_chunks} chunks saved.")
print(f"TXT saved to: {txt_all_path}")
print(f"JSONL saved to: {json_all_path}")
