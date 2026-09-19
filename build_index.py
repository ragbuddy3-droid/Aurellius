import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, CHUNK_JSONL_PATH, INDEX_PATH, STORE_PATH

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_source_manifest(path: str) -> List[str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest source tidak ditemukan: {manifest_path}")

    if manifest_path.suffix.lower() == ".json":
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        values = data.get("sources", []) if isinstance(data, dict) else data
        if not isinstance(values, list):
            raise RuntimeError("Format manifest JSON harus berupa list atau object dengan key 'sources'.")
        return [str(item).strip().lower() for item in values if str(item).strip()]

    return [
        line.strip().lower()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def is_chunk_allowed(chunk: dict, source_filters: Optional[List[str]]) -> bool:
    if not source_filters:
        return True

    source_pdf = str(chunk.get("source_pdf", "")).lower()
    source_book = str(chunk.get("source_book", "")).lower()
    return any(filter_value in source_pdf or filter_value in source_book for filter_value in source_filters)

def build_faiss_index(
    jsonl_path: str,
    index_path: str,
    store_path: str,
    source_filters: Optional[List[str]] = None,
):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Chunk file not found: {os.path.abspath(jsonl_path)}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = load_jsonl(jsonl_path)
    chunks = [chunk for chunk in chunks if is_chunk_allowed(chunk, source_filters)]
    if not chunks:
        raise RuntimeError("Tidak ada chunk yang cocok dengan filter sumber/index.")

    texts = ["passage: " + c["content"].strip() for c in chunks]

    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=False
    )
    emb = np.array(emb, dtype="float32")
    emb = l2_normalize(emb)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(emb)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Done. Indexed {len(chunks)} chunks.")
    print(f"Saved: {index_path} and {store_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS index dari hasil chunk. Bisa dibuat terpisah per penelitian dengan source filter."
    )
    parser.add_argument("--chunks", default=CHUNK_JSONL_PATH, help="Path chunks JSONL input.")
    parser.add_argument("--index", default=INDEX_PATH, help="Path output FAISS index.")
    parser.add_argument("--store", default=STORE_PATH, help="Path output metadata store JSON.")
    parser.add_argument(
        "--source-subdir",
        default="",
        help="Hanya index chunk yang source_pdf/source_book-nya mengandung teks ini, misalnya context_injection.",
    )
    parser.add_argument(
        "--source-manifest",
        default="",
        help="Path JSON/TXT whitelist source_book/source_pdf untuk index penelitian tertentu.",
    )
    args = parser.parse_args()

    filters: Optional[List[str]] = None
    if args.source_subdir.strip():
        filters = [args.source_subdir.strip().lower()]
    elif args.source_manifest.strip():
        filters = load_source_manifest(args.source_manifest)

    if filters:
        print(f"Filter index aktif: {filters}")

    build_faiss_index(
        jsonl_path=args.chunks,
        index_path=args.index,
        store_path=args.store,
        source_filters=filters,
    )
