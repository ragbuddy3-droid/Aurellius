import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from chatbot import generate_answer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GT_PATH = PROJECT_DIR / "evaluation" / "ground_truth_master.csv"
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"


def load_results_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_ground_truth_rows(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "File ground truth berbentuk Excel membutuhkan pandas dan openpyxl. "
                "Install dulu dependency evaluasi."
            ) from exc

        df = pd.read_excel(path)
        return df.fillna("").to_dict(orient="records")

    raise ValueError(f"Format file tidak didukung: {path}")


def resolve_allowed_grades(grade_value: Any) -> List[int]:
    try:
        grade = int(str(grade_value).strip())
    except (TypeError, ValueError):
        return [4, 5, 6]

    if grade <= 4:
        return [4]
    if grade == 5:
        return [4, 5]
    return [4, 5, 6]


def parse_allowed_grades(raw_value: Any, fallback_grade: Any) -> List[int]:
    if raw_value not in (None, ""):
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return resolve_allowed_grades(fallback_grade)


def chunk_to_context_text(chunk: Dict[str, Any]) -> str:
    heading = " | ".join(str(h) for h in (chunk.get("headings") or []))
    content = (chunk.get("content") or "").strip()
    source_book = chunk.get("source_book") or "-"
    if heading:
        return f"Sumber buku: {source_book}\nHeading: {heading}\nIsi: {content}"
    return f"Sumber buku: {source_book}\nIsi: {content}"


def build_result_row(row: Dict[str, Any], answer: str, chunks: List[Dict[str, Any]], latency: float) -> Dict[str, Any]:
    retrieved_contexts = [chunk_to_context_text(chunk) for chunk in chunks]
    retrieved_chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
    retrieved_books = [chunk.get("source_book", "") for chunk in chunks]

    return {
        "id": row.get("id", ""),
        "subject": row.get("subject", ""),
        "grade": row.get("grade", ""),
        "book_code": row.get("book_code", ""),
        "question": row.get("question", ""),
        "ground_truth": row.get("ground_truth", ""),
        "reference_chunks": row.get("reference_chunks", ""),
        "allowed_grades": json.dumps(resolve_allowed_grades(row.get("grade")), ensure_ascii=False),
        "response": answer,
        "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids, ensure_ascii=False),
        "retrieved_books": json.dumps(retrieved_books, ensure_ascii=False),
        "retrieved_contexts": json.dumps(retrieved_contexts, ensure_ascii=False),
        "latency_seconds": f"{latency:.2f}",
    }


def run_generation(rows: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    results = []
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        question = (row.get("question") or "").strip()
        allowed_grades = resolve_allowed_grades(row.get("grade"))
        started = time.perf_counter()

        try:
            answer, chunks = generate_answer(
                question=question,
                allowed_grades=allowed_grades,
                provider=provider,
            )
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            chunks = []

        latency = time.perf_counter() - started
        result_row = build_result_row(row, answer, chunks, latency)
        results.append(result_row)
        print(f"[{idx}/{total}] selesai: {question} ({latency:.2f}s)")

    return results


def response_has_error(response: Any) -> bool:
    text = str(response or "").strip()
    return text.startswith("[ERROR]")


def rerun_failed_rows(existing_rows: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    updated_rows: List[Dict[str, Any]] = []
    total = len(existing_rows)
    failed_indexes = [idx for idx, row in enumerate(existing_rows, start=1) if response_has_error(row.get("response"))]

    print(f"Menemukan {len(failed_indexes)} baris error yang akan dicoba ulang.")

    for idx, row in enumerate(existing_rows, start=1):
        if not response_has_error(row.get("response")):
            updated_rows.append(dict(row))
            continue

        question = (row.get("question") or "").strip()
        allowed_grades = parse_allowed_grades(row.get("allowed_grades"), row.get("grade"))
        started = time.perf_counter()

        try:
            answer, chunks = generate_answer(
                question=question,
                allowed_grades=allowed_grades,
                provider=provider,
            )
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            chunks = []

        latency = time.perf_counter() - started
        repaired_row = build_result_row(row, answer, chunks, latency)
        updated_rows.append(repaired_row)
        print(f"[retry {idx}/{total}] selesai: {question} ({latency:.2f}s)")

    return updated_rows


def save_results_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


DECIMAL_RE = re.compile(r"^-?\d+\.\d+$")


def to_excel_friendly_value(value: Any) -> Any:
    if isinstance(value, float):
        return str(value).replace(".", ",")
    if isinstance(value, str):
        text = value.strip()
        if DECIMAL_RE.match(text):
            return text.replace(".", ",")
    return value


def make_excel_friendly_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: to_excel_friendly_value(value) for key, value in row.items()}
        for row in rows
    ]


def save_results_excel_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_results_xlsx(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Simpan ke Excel membutuhkan pandas dan openpyxl. Install requirements-eval.txt terlebih dahulu."
        ) from exc

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name="evaluation_results")


def save_results_excel_xlsx(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Simpan ke Excel membutuhkan pandas dan openpyxl. Install requirements-eval.txt terlebih dahulu."
        ) from exc

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name="evaluation_results_excel")


def compute_bertscore(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import torch
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore belum bisa dijalankan karena package bert-score belum terpasang."
        ) from exc

    candidates = [row["response"] for row in rows]
    references = [row["ground_truth"] for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    precision, recall, f1 = bert_score(
        candidates,
        references,
        lang="id",
        device=device,
        verbose=True,
    )

    for row, p, r, f in zip(rows, precision, recall, f1):
        row["bertscore_precision"] = f"{float(p):.4f}"
        row["bertscore_recall"] = f"{float(r):.4f}"
        row["bertscore_f1"] = f"{float(f):.4f}"

    summary = {
        "metric": "BERTScore",
        "device": device,
        "avg_precision": round(mean(float(x) for x in precision), 4),
        "avg_recall": round(mean(float(x) for x in recall), 4),
        "avg_f1": round(mean(float(x) for x in f1), 4),
    }
    return rows, summary


def export_ragas_input(rows: List[Dict[str, Any]], path: Path):
    payload = []
    for row in rows:
        payload.append(
            {
                "user_input": row["question"],
                "response": row["response"],
                "reference": row["ground_truth"],
                "retrieved_contexts": json.loads(row["retrieved_contexts"]),
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def try_run_ragas(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, str | None]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
    except ImportError:
        return rows, None, "Dependency RAGAS belum terpasang, jadi perhitungan RAGAS dilewati."

    dataset_rows = []
    for row in rows:
        dataset_rows.append(
            {
                "question": row["question"],
                "answer": row["response"],
                "ground_truth": row["ground_truth"],
                "contexts": json.loads(row["retrieved_contexts"]),
            }
        )

    try:
        dataset = Dataset.from_list(dataset_rows)
        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy, faithfulness, answer_correctness],
        )
    except Exception as exc:
        return rows, None, f"RAGAS belum berhasil dijalankan: {exc}"

    try:
        result_df = result.to_pandas()
        score_rows = result_df.to_dict(orient="records")
    except Exception:
        return rows, None, "RAGAS berhasil dipanggil, tetapi format hasil tidak bisa dibaca otomatis."

    for row, score_row in zip(rows, score_rows):
        for key, value in score_row.items():
            if key in {"question", "answer", "ground_truth", "contexts"}:
                continue
            row[f"ragas_{key}"] = "" if value is None else f"{float(value):.4f}"

    summary = {}
    numeric_keys = [key for key in score_rows[0].keys() if key not in {"question", "answer", "ground_truth", "contexts"}]
    for key in numeric_keys:
        values = [score_row.get(key) for score_row in score_rows if isinstance(score_row.get(key), (int, float))]
        if values:
            summary[f"avg_{key}"] = round(mean(values), 4)

    return rows, summary or None, None


def save_json(data: Dict[str, Any], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluasi ground truth chatbot dengan BERTScore dan input RAGAS.")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT_PATH), help="Path ke file ground truth master (.csv atau .xlsx).")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "qwen"], help="Model generator jawaban.")
    parser.add_argument("--skip-bertscore", action="store_true", help="Lewati perhitungan BERTScore.")
    parser.add_argument("--run-ragas", action="store_true", help="Coba jalankan perhitungan RAGAS jika dependency dan konfigurasi sudah siap.")
    parser.add_argument("--resume-results", help="Path ke CSV hasil evaluasi sebelumnya. Jika diisi, script hanya akan mencoba ulang baris yang masih [ERROR].")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume_results:
        previous_results_path = Path(args.resume_results)
        generated_rows = rerun_failed_rows(load_results_rows(previous_results_path), provider=args.provider)
        ground_truth_path = Path(args.ground_truth)
        source_results_path = str(previous_results_path)
    else:
        ground_truth_path = Path(args.ground_truth)
        rows = load_ground_truth_rows(ground_truth_path)
        if not rows:
            raise RuntimeError("Ground truth kosong.")
        print(f"Memulai evaluasi dengan provider: {args.provider}")
        generated_rows = run_generation(rows, provider=args.provider)
        source_results_path = None

    ragas_input_path = RESULTS_DIR / f"ragas_input_{args.provider}_{timestamp}.json"
    export_ragas_input(generated_rows, ragas_input_path)

    summary: Dict[str, Any] = {
        "provider": args.provider,
        "total_questions": len(generated_rows),
        "ground_truth_path": str(ground_truth_path),
        "ragas_input_path": str(ragas_input_path),
        "avg_latency_seconds": round(mean(float(row["latency_seconds"]) for row in generated_rows), 4),
    }
    if source_results_path:
        summary["resume_results_path"] = source_results_path

    notes: List[str] = []

    if not args.skip_bertscore:
        try:
            generated_rows, bert_summary = compute_bertscore(generated_rows)
            summary["bertscore"] = bert_summary
        except Exception as exc:
            notes.append(str(exc))

    if args.run_ragas:
        generated_rows, ragas_summary, ragas_note = try_run_ragas(generated_rows)
        if ragas_summary:
            summary["ragas"] = ragas_summary
        if ragas_note:
            notes.append(ragas_note)

    if notes:
        summary["notes"] = notes

    results_csv_path = RESULTS_DIR / f"evaluation_results_{args.provider}_{timestamp}.csv"
    results_xlsx_path = RESULTS_DIR / f"evaluation_results_{args.provider}_{timestamp}.xlsx"
    results_excel_csv_path = RESULTS_DIR / f"evaluation_results_{args.provider}_{timestamp}_excel.csv"
    results_excel_xlsx_path = RESULTS_DIR / f"evaluation_results_{args.provider}_{timestamp}_excel.xlsx"
    summary_json_path = RESULTS_DIR / f"evaluation_summary_{args.provider}_{timestamp}.json"

    save_results_csv(generated_rows, results_csv_path)
    save_results_xlsx(generated_rows, results_xlsx_path)
    excel_friendly_rows = make_excel_friendly_rows(generated_rows)
    save_results_excel_csv(excel_friendly_rows, results_excel_csv_path)
    save_results_excel_xlsx(excel_friendly_rows, results_excel_xlsx_path)
    save_json(summary, summary_json_path)

    print(
        f"\nHasil evaluasi tersimpan di:\n"
        f"- {results_csv_path}\n"
        f"- {results_xlsx_path}\n"
        f"- {results_excel_csv_path}\n"
        f"- {results_excel_xlsx_path}\n"
        f"- {summary_json_path}\n"
        f"- {ragas_input_path}"
    )


if __name__ == "__main__":
    main()
