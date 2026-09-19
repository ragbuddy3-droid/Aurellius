import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from chatbot import generate_answer


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
DEFAULT_INPUT_PATH = Path(r"C:\Users\aurel\Downloads\off topic.xlsx")
QUESTION_COLUMN_CANDIDATES = ["question", "pertanyaan", "query", "prompt"]
SUPPORTED_PROVIDERS = ["gemini", "qwen"]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "File Excel membutuhkan pandas dan openpyxl. Install dependency evaluasi terlebih dahulu."
            ) from exc

        df = pd.read_excel(path)
        return df.fillna("").to_dict(orient="records")

    raise ValueError(f"Format file tidak didukung: {path}")


def detect_question_key(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        raise RuntimeError("File pertanyaan off-topic kosong.")

    first_row = rows[0]
    normalized_map = {str(key).strip().lower(): key for key in first_row.keys()}
    for candidate in QUESTION_COLUMN_CANDIDATES:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for key in first_row.keys():
        value = str(first_row.get(key, "")).strip()
        if value:
            return key

    raise RuntimeError("Tidak menemukan kolom pertanyaan yang bisa dipakai.")


def chunk_to_context_text(chunk: Dict[str, Any]) -> str:
    heading = " | ".join(str(h) for h in (chunk.get("headings") or []))
    content = (chunk.get("content") or "").strip()
    source_book = chunk.get("source_book") or "-"
    chunk_id = chunk.get("chunk_id") or "-"

    lines = [f"Chunk ID: {chunk_id}", f"Sumber buku: {source_book}"]
    if heading:
        lines.append(f"Heading: {heading}")
    lines.append("Isi:")
    lines.append(content)
    return "\n".join(lines)


def stringify_contexts(chunks: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    books = [str(chunk.get("source_book", "")) for chunk in chunks]
    contexts = [chunk_to_context_text(chunk) for chunk in chunks]

    return (
        json.dumps(chunk_ids, ensure_ascii=False),
        json.dumps(books, ensure_ascii=False),
        json.dumps(contexts, ensure_ascii=False),
    )


def build_result_row(
    provider: str,
    row_index: int,
    question: str,
    source_row: Dict[str, Any],
    answer: str,
    chunks: List[Dict[str, Any]],
    latency: float,
) -> Dict[str, Any]:
    retrieved_chunk_ids, retrieved_books, retrieved_contexts = stringify_contexts(chunks)
    retrieved_contexts_text = "\n\n" + ("\n\n".join(chunk_to_context_text(chunk) for chunk in chunks) if chunks else "")

    return {
        "row_no": row_index,
        "provider": provider,
        "question": question,
        "response": answer,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_books": retrieved_books,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_contexts_text": retrieved_contexts_text.strip(),
        "retrieved_count": len(chunks),
        "latency_seconds": f"{latency:.2f}",
        "source_row_json": json.dumps(source_row, ensure_ascii=False),
    }


def run_provider(rows: List[Dict[str, Any]], question_key: str, provider: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        question = str(row.get(question_key, "")).strip()
        started = time.perf_counter()

        try:
            answer, chunks = generate_answer(question=question, provider=provider)
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            chunks = []

        latency = time.perf_counter() - started
        result_row = build_result_row(
            provider=provider,
            row_index=idx,
            question=question,
            source_row=row,
            answer=answer,
            chunks=chunks,
            latency=latency,
        )
        results.append(result_row)
        print(f"[{provider} {idx}/{total}] selesai: {question} ({latency:.2f}s)")

    return results


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_xlsx(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Simpan ke Excel membutuhkan pandas dan openpyxl. Install dependency evaluasi terlebih dahulu."
        ) from exc

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name="off_topic_results")


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_provider_outputs(provider: str, rows: List[Dict[str, Any]], timestamp: str) -> Tuple[Path, Path, Path]:
    csv_path = RESULTS_DIR / f"off_topic_results_{provider}_{timestamp}.csv"
    xlsx_path = RESULTS_DIR / f"off_topic_results_{provider}_{timestamp}.xlsx"
    summary_path = RESULTS_DIR / f"off_topic_summary_{provider}_{timestamp}.json"

    save_csv(rows, csv_path)
    save_xlsx(rows, xlsx_path)
    provider_summary = {
        "provider": provider,
        "total_rows": len(rows),
        "avg_latency_seconds": round(mean(float(row["latency_seconds"]) for row in rows), 4) if rows else 0.0,
    }
    save_json(provider_summary, summary_path)
    return csv_path, xlsx_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jalankan pertanyaan off-topic dari Excel/CSV dan simpan jawaban beserta context retrieval."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path ke file pertanyaan off-topic (.xlsx, .xls, .csv).",
    )
    parser.add_argument(
        "--provider",
        default="both",
        choices=["gemini", "qwen", "both"],
        help="Provider yang dipakai untuk menjawab pertanyaan.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path)
    question_key = detect_question_key(rows)
    providers = SUPPORTED_PROVIDERS if args.provider == "both" else [args.provider]

    all_results: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "input_path": str(input_path),
        "question_column": question_key,
        "total_questions": len(rows),
        "providers": {},
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for provider in providers:
        provider_results = run_provider(rows, question_key=question_key, provider=provider)
        all_results.extend(provider_results)
        summary["providers"][provider] = {
            "total_rows": len(provider_results),
            "avg_latency_seconds": round(mean(float(row["latency_seconds"]) for row in provider_results), 4),
        }
        provider_csv_path, provider_xlsx_path, provider_summary_path = save_provider_outputs(
            provider=provider,
            rows=provider_results,
            timestamp=timestamp,
        )
        print(
            f"\nHasil {provider} tersimpan di:\n"
            f"- {provider_csv_path}\n"
            f"- {provider_xlsx_path}\n"
            f"- {provider_summary_path}\n"
        )

    csv_path = RESULTS_DIR / f"off_topic_results_{args.provider}_{timestamp}.csv"
    xlsx_path = RESULTS_DIR / f"off_topic_results_{args.provider}_{timestamp}.xlsx"
    summary_path = RESULTS_DIR / f"off_topic_summary_{args.provider}_{timestamp}.json"

    save_csv(all_results, csv_path)
    save_xlsx(all_results, xlsx_path)
    save_json(summary, summary_path)

    print(
        "\nHasil off-topic tersimpan di:\n"
        f"- {csv_path}\n"
        f"- {xlsx_path}\n"
        f"- {summary_path}"
    )


if __name__ == "__main__":
    main()
