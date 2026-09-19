import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from datasets import Dataset

from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, GEMINI_MODEL_NAME


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("File input RAGAS kosong atau formatnya tidak valid.")
    return rows


def build_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "user_input": row.get("user_input", ""),
                "response": row.get("response", ""),
                "reference": row.get("reference", ""),
                "retrieved_contexts": row.get("retrieved_contexts", []),
            }
        )
    return Dataset.from_list(normalized)


def get_metrics(llm_wrapper, embeddings_wrapper):
    try:
        from ragas.metrics import answer_relevancy, faithfulness
        try:
            from ragas.metrics import context_relevance as context_metric
        except ImportError:
            from ragas.metrics import ContextRelevance as context_metric

        prepared = []
        for metric in [faithfulness, answer_relevancy, context_metric]:
            metric_instance = metric() if callable(metric) and not hasattr(metric, "score") else metric
            if hasattr(metric_instance, "llm"):
                metric_instance.llm = llm_wrapper
            if hasattr(metric_instance, "embeddings"):
                metric_instance.embeddings = embeddings_wrapper
            prepared.append(metric_instance)
        return prepared
    except ImportError as exc:
        raise RuntimeError(
            "Versi ragas yang terpasang belum menyediakan metric context relevance. "
            "Gunakan versi ragas yang mendukung context_relevance/ContextRelevance agar hasil evaluasi "
            "sesuai dengan teori yang kamu pakai."
        ) from exc


def build_gemini_llm_wrapper():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY belum diset untuk evaluator RAGAS.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Dependency evaluator Gemini untuk RAGAS belum lengkap. "
            "Install dulu requirements-eval.txt."
        ) from exc

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def build_embeddings_wrapper():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Dependency embedding untuk RAGAS belum lengkap. Install dulu requirements-eval.txt."
        ) from exc

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def dataframe_rows_from_result(result) -> List[Dict[str, Any]]:
    try:
        df = result.to_pandas()
        return df.to_dict(orient="records")
    except Exception as exc:
        raise RuntimeError(f"Hasil RAGAS tidak bisa dikonversi ke tabel: {exc}") from exc


def save_csv(rows: List[Dict[str, Any]], path: Path):
    import csv

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


def save_excel_csv(rows: List[Dict[str, Any]], path: Path):
    import csv

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_xlsx(rows: List[Dict[str, Any]], path: Path):
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
        df.to_excel(writer, index=False, sheet_name="ragas_scores")


def save_excel_xlsx(rows: List[Dict[str, Any]], path: Path):
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
        df.to_excel(writer, index=False, sheet_name="ragas_scores_excel")


def save_json(data: Dict[str, Any], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    if not rows:
        return summary

    numeric_keys = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float)):
            numeric_keys.append(key)

    if not numeric_keys:
        for key in rows[0].keys():
            values = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    numeric_value = float(value)
                    if not math.isnan(numeric_value):
                        values.append(numeric_value)
                else:
                    try:
                        numeric_value = float(value)
                        if not math.isnan(numeric_value):
                            values.append(numeric_value)
                    except (TypeError, ValueError):
                        pass
            if values:
                summary[f"avg_{key}"] = round(mean(values), 4)
        return summary

    for key in numeric_keys:
        values = []
        for row in rows:
            if isinstance(row.get(key), (int, float)):
                numeric_value = float(row[key])
                if not math.isnan(numeric_value):
                    values.append(numeric_value)
        if values:
            summary[f"avg_{key}"] = round(mean(values), 4)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Jalankan evaluasi RAGAS dari file ragas_input JSON.")
    parser.add_argument("--input", required=True, help="Path ke file ragas_input_*.json")
    parser.add_argument("--judge", default="gemini", choices=["gemini"], help="LLM evaluator untuk RAGAS.")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_rows(input_path)
    dataset = build_dataset(rows)
    if args.judge == "gemini":
        llm_wrapper = build_gemini_llm_wrapper()
    else:
        raise ValueError(f"Judge tidak didukung: {args.judge}")

    embeddings_wrapper = build_embeddings_wrapper()
    metrics = get_metrics(llm_wrapper, embeddings_wrapper)

    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError("Package ragas belum terpasang.") from exc

    print(f"Menjalankan RAGAS untuk {len(rows)} sampel dengan judge: {args.judge}")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
    )
    score_rows = dataframe_rows_from_result(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_path.stem.replace("ragas_input_", "")

    results_csv_path = RESULTS_DIR / f"ragas_scores_{stem}_{timestamp}.csv"
    results_xlsx_path = RESULTS_DIR / f"ragas_scores_{stem}_{timestamp}.xlsx"
    results_excel_csv_path = RESULTS_DIR / f"ragas_scores_{stem}_{timestamp}_excel.csv"
    results_excel_xlsx_path = RESULTS_DIR / f"ragas_scores_{stem}_{timestamp}_excel.xlsx"
    summary_json_path = RESULTS_DIR / f"ragas_summary_{stem}_{timestamp}.json"

    save_csv(score_rows, results_csv_path)
    save_xlsx(score_rows, results_xlsx_path)
    excel_friendly_rows = make_excel_friendly_rows(score_rows)
    save_excel_csv(excel_friendly_rows, results_excel_csv_path)
    save_excel_xlsx(excel_friendly_rows, results_excel_xlsx_path)
    summary = {
        "input_path": str(input_path),
        "judge": args.judge,
        "total_questions": len(rows),
        "metrics": summarize_scores(score_rows),
    }
    save_json(summary, summary_json_path)

    print(
        f"\nHasil RAGAS tersimpan di:\n"
        f"- {results_csv_path}\n"
        f"- {results_xlsx_path}\n"
        f"- {results_excel_csv_path}\n"
        f"- {results_excel_xlsx_path}\n"
        f"- {summary_json_path}"
    )


if __name__ == "__main__":
    main()
