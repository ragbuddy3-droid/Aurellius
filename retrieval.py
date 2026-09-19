import json
import os
import re
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    INDEX_PATH,
    LEGACY_INDEX_PATH,
    LEGACY_STORE_PATH,
    PRE_K,
    RETRIEVAL_DEBUG,
    RETRIEVAL_MIN_SCORE,
    STORE_PATH,
    TOP_K,
)

INDEX_CANDIDATES = [INDEX_PATH, LEGACY_INDEX_PATH]
STORE_CANDIDATES = [STORE_PATH, LEGACY_STORE_PATH]
_EMBED_MODEL = None
_FAISS_INDEX = None
_STORE_DATA = None
_ACTIVE_INDEX_PATH = INDEX_PATH
_ACTIVE_STORE_PATH = STORE_PATH

GRADE_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
}

ROMAN_CHAPTER_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def dedup_list(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def load_store(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Store not found: {os.path.abspath(path)}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_existing_path(candidates: List[str], label: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    joined = ", ".join(os.path.abspath(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found. Expected one of: {joined}")


def configure_retrieval_paths(index_path: Optional[str] = None, store_path: Optional[str] = None) -> None:
    global _ACTIVE_INDEX_PATH, _ACTIVE_STORE_PATH, _FAISS_INDEX, _STORE_DATA

    if index_path:
        _ACTIVE_INDEX_PATH = index_path
    if store_path:
        _ACTIVE_STORE_PATH = store_path

    # Reset cache supaya retrieval berikutnya benar-benar memakai index/store baru.
    _FAISS_INDEX = None
    _STORE_DATA = None


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    return _EMBED_MODEL


def get_faiss_index():
    global _FAISS_INDEX
    if _FAISS_INDEX is None:
        index_path = resolve_existing_path([_ACTIVE_INDEX_PATH, LEGACY_INDEX_PATH], "FAISS index")
        _FAISS_INDEX = faiss.read_index(index_path)
    return _FAISS_INDEX


def get_store_data() -> List[Dict]:
    global _STORE_DATA
    if _STORE_DATA is None:
        store_path = resolve_existing_path([_ACTIVE_STORE_PATH, LEGACY_STORE_PATH], "Metadata store")
        _STORE_DATA = load_store(store_path)
    return _STORE_DATA

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    q = "query: " + q.strip()
    emb = model.encode([q], normalize_embeddings=False)
    emb = np.array(emb, dtype="float32")
    return l2_normalize(emb)

def parse_ch_number(chunk_id: str) -> Optional[int]:
    m = re.search(r"CH(\d+)", chunk_id or "")
    return int(m.group(1)) if m else None

def extract_first_heading(chunk: Dict) -> str:
    hs = chunk.get("headings") or []
    return str(hs[0]) if hs else ""


def build_debug_candidate(chunk: Dict, raw_score: float, final_score: Optional[float] = None) -> Dict:
    return {
        "chunk_id": chunk.get("chunk_id", "-"),
        "source_book": chunk.get("source_book", "-"),
        "heading": extract_first_heading(chunk) or "-",
        "raw_score": raw_score,
        "final_score": final_score,
        "categories": dedup_list(chunk.get("categories", [])),
        "cue_patterns": dedup_list(chunk.get("cue_patterns", [])),
    }


def format_debug_candidate_line(rank: int, item: Dict, include_final: bool = False) -> str:
    base = (
        f"[{rank}] {item.get('chunk_id')} | {item.get('source_book')} | "
        f"raw={item.get('raw_score', 0.0):.4f} | heading={item.get('heading', '-')}"
    )
    if include_final:
        base += f" | final={item.get('final_score', 0.0):.4f}"
    cats = ", ".join(item.get("categories", [])) or "-"
    cues = ", ".join(item.get("cue_patterns", [])) or "-"
    return f"{base} | categories={cats} | cues={cues}"


def debug_print_retrieval(
    question: str,
    pre_k: int,
    top_k: int,
    threshold: float,
    pre_candidates: List[Dict],
    filtered_out: List[Dict],
    kept_candidates: List[Dict],
    final_chunks: List[Dict],
    allowed_grades: Optional[List[int]],
    requested_chapter: Optional[int],
    allow_glossary: bool,
    allow_instructions: bool,
) -> None:
    print("\n===== DEBUG RETRIEVAL =====")
    print(f"Pertanyaan : {question}")
    print(f"pre_k      : {pre_k}")
    print(f"top_k      : {top_k}")
    print(f"threshold  : {threshold:.2f}")
    print(f"allowed_grades : {allowed_grades if allowed_grades else 'semua'}")
    print(f"chapter_filter : {requested_chapter if requested_chapter is not None else 'tidak ada'}")
    print(f"allow_glossary : {allow_glossary}")
    print(f"allow_steps    : {allow_instructions}")

    print("\n-- Kandidat PRE_K dari FAISS --")
    if not pre_candidates:
        print("(tidak ada kandidat awal)")
    else:
        for i, item in enumerate(pre_candidates, start=1):
            print(format_debug_candidate_line(i, item))

    print("\n-- Kandidat Gugur Saat Filtering --")
    if not filtered_out:
        print("(tidak ada kandidat yang gugur)")
    else:
        for i, item in enumerate(filtered_out, start=1):
            print(f"{format_debug_candidate_line(i, item)} | reason={item.get('reason', '-')}")

    print("\n-- Kandidat Lolos Filtering / Reranking --")
    if not kept_candidates:
        print("(tidak ada kandidat yang lolos)")
    else:
        for i, item in enumerate(kept_candidates, start=1):
            print(format_debug_candidate_line(i, item, include_final=True))

    if not final_chunks:
        print("\nHasil      : tidak ada chunk yang lolos threshold/filter.")
        print("===========================\n")
        return

    print("\n-- TOP_K Final --")
    for i, chunk in enumerate(final_chunks, start=1):
        heading = extract_first_heading(chunk) or "-"
        raw_score = chunk.get("_raw_score")
        final_score = chunk.get("_final_score")
        print(
            f"[{i}] {chunk.get('chunk_id')} | {chunk.get('source_book')} | "
            f"raw={raw_score:.4f} | final={final_score:.4f} | heading={heading}"
        )

    print("===========================\n")

def contains_word(text: str, word: str) -> bool:
    return word.lower() in (text or "").lower()

def question_wants_list(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["apa saja", "sifat-sifat", "sifat sifat", "jenis-jenis", "jenis jenis", "macam-macam", "macam macam", "daftar", "sebutkan"])

def question_wants_steps(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["cara", "langkah", "percobaan", "praktik", "praktikum", "lakukan", "buatlah", "kerjakan"])

def question_is_glossary(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["arti", "makna", "maksud kata", "maksud istilah", "definisi istilah", "glosarium"])


def question_asks_definition(question: str) -> bool:
    q = (question or "").lower().strip()
    return (
        q.startswith("apa itu ")
        or q.startswith("apakah itu ")
        or q.endswith(" adalah")
        or "pengertian" in q
        or "definisi" in q
    )

def is_glossary_chunk(chunk: Dict) -> bool:
    cats = set(chunk.get("categories", []))
    cues = set(chunk.get("cue_patterns", []))
    return ("GLOSARIUM" in cats) or ("GLOSSARY_BOX" in cues)

def is_instruction_chunk(chunk: Dict) -> bool:
    cats = set(chunk.get("categories", []))
    cues = set(chunk.get("cue_patterns", []))
    return ("INSTRUKSI" in cats) or ("IMPERATIVE_TASK" in cues)

def is_concept_chunk(chunk: Dict) -> bool:
    cats = set(chunk.get("categories", []))
    return "KONSEP" in cats

def numbered_heading_value(chunk: Dict) -> Optional[int]:
    """
    If heading is like "1. ..." return 1, etc.
    """
    h = extract_first_heading(chunk).strip()
    m = re.match(r"^(\d+)\.", h)
    return int(m.group(1)) if m else None

def prefer_1_to_5(chunk: Dict) -> bool:
    n = numbered_heading_value(chunk)
    return n is not None and 1 <= n <= 5

def extract_headings_text(chunk: Dict) -> str:
    parts = [str(h) for h in (chunk.get("headings") or [])]
    chapter = chunk.get("chapter")
    if chapter:
        parts.append(str(chapter))
    return " ".join(parts)


def extract_question_chapter_number(question: str) -> Optional[int]:
    m = re.search(r"\bBab\s+([0-9IVX]+)\b", question or "", flags=re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).upper()
    if raw.isdigit():
        return int(raw)
    return ROMAN_CHAPTER_MAP.get(raw)


def chunk_chapter_number(chunk: Dict) -> Optional[int]:
    value = chunk.get("chapter_number")
    if isinstance(value, int):
        return value
    try:
        if value is not None and str(value).strip() != "":
            numeric_value = float(value)
            if not math.isnan(numeric_value):
                return int(numeric_value)
    except (TypeError, ValueError):
        pass

    for heading in chunk.get("headings") or []:
        m = re.search(r"\bBab\s+([0-9IVX]+)\b", str(heading), flags=re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1).upper()
        if raw.isdigit():
            return int(raw)
        return ROMAN_CHAPTER_MAP.get(raw)
    return None


def is_chapter_allowed(chunk: Dict, requested_chapter: Optional[int]) -> bool:
    if requested_chapter is None:
        return True
    chunk_number = chunk_chapter_number(chunk)
    if chunk_number is None:
        return True
    return chunk_number == requested_chapter


def extract_book_grade(source_book: str) -> Optional[int]:
    text = (source_book or "").upper()
    if not text:
        return None

    normalized = re.sub(r"[^A-Z0-9]+", "_", text)
    parts = [part for part in normalized.split("_") if part]

    for part in reversed(parts):
        if part in GRADE_MAP:
            return GRADE_MAP[part]
        if part.isdigit():
            return int(part)

    compact_patterns = [
        (r"KLS(III|II|IV|VI|V|I)\b", 1),
        (r"KELAS(III|II|IV|VI|V|I)\b", 1),
    ]
    for pattern, group_idx in compact_patterns:
        match = re.search(pattern, normalized)
        if match:
            return GRADE_MAP.get(match.group(group_idx))

    return None


def is_grade_allowed(chunk: Dict, allowed_grades: Optional[List[int]]) -> bool:
    if not allowed_grades:
        return True
    chunk_grade = extract_book_grade(chunk.get("source_book", ""))
    if chunk_grade is None:
        return True
    return chunk_grade in allowed_grades


def is_source_allowed(chunk: Dict, source_path_filters: Optional[List[str]]) -> bool:
    if not source_path_filters:
        return True

    source_pdf = str(chunk.get("source_pdf", "")).lower()
    source_book = str(chunk.get("source_book", "")).lower()
    normalized_filters = [item.strip().lower() for item in source_path_filters if str(item).strip()]
    if not normalized_filters:
        return True

    return any(
        filter_value in source_pdf or filter_value in source_book
        for filter_value in normalized_filters
    )

#yg bkl di buang
STOPWORDS = {
    "apa", "apakah", "itu", "ini", "dan", "yang", "di", "ke", "dari", "untuk",
    "pada", "adalah", "kita", "bisa", "bagi", "dengan", "dalam", "karena",
    "atau", "seperti", "saat", "agar", "jadi", "mengapa", "kenapa", "bagaimana",
    "jelaskan", "sebutkan",
}

def query_keywords(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Zà-ÿ]+", (question or "").lower())
    return [w for w in words if len(w) >= 4 and w not in STOPWORDS]

def keyword_overlap_score(question: str, chunk: Dict) -> float:
    keys = query_keywords(question)
    if not keys:
        return 0.0

    text = (extract_headings_text(chunk) + " " + (chunk.get("content") or "")).lower()
    matched = sum(1 for key in keys if key in text)
    if matched == 0:
        return 0.0

    return min(0.18, matched * 0.06)

def score_adjust(question: str, chunk: Dict) -> float:
    q = question.lower()
    cues = set(chunk.get("cue_patterns", []))
    cats = set(chunk.get("categories", []))
    headings_text = extract_headings_text(chunk).lower()
    content_text = (chunk.get("content") or "").lower()

    adj = 0.0

    if question_wants_list(question):
        if prefer_1_to_5(chunk):
            adj += 0.18
        elif numbered_heading_value(chunk) is not None:
            adj += 0.06  
        if "PROMPT_INTRO" in cues:
            adj -= 0.12  
        if "INSTRUKSI" in cats:
            adj -= 0.18
        if "EVALUASI" in cats:
            adj -= 0.12
        if "SECTION" in cats:
            adj -= 0.08
        if is_concept_chunk(chunk):
            adj += 0.08

    if question_wants_list(question) and ("DEFINITION" in cues or "CAUSE_EFFECT" in cues or "FACT_EXPLANATION" in cues or "EXAMPLE_ILLUSTRATION" in cues):
        adj += 0.06

    if "KONSEP" in cats:
        adj += 0.03

    adj += keyword_overlap_score(question, chunk)

    if question_asks_definition(question):
        if "INSTRUKSI" in cats or "EVALUASI" in cats:
            adj -= 0.14

    if any(k in q for k in ["mengapa", "kenapa", "sebab", "akibat", "karena", "bagaimana bisa"]):
        if "CAUSE_EFFECT" in cues:
            adj += 0.08

    if any(k in q for k in ["contoh", "misalnya", "ilustrasi"]):
        if "EXAMPLE_ILLUSTRATION" in cues:
            adj += 0.08

    return adj


def expand_by_ch_range(store: List[Dict], source_book: str, start_ch: int, end_ch: int) -> List[Dict]:
    out = []
    for c in store:
        if c.get("source_book") != source_book:
            continue
        n = parse_ch_number(c.get("chunk_id"))
        if n is None:
            continue
        if start_ch <= n <= end_ch:
            out.append(c)
    out.sort(key=lambda x: (parse_ch_number(x.get("chunk_id")) or 10**9))
    return out


#Main Retrieval
def retrieve(
    question: str,
    top_k: int = TOP_K,
    pre_k: int = PRE_K,
    allowed_grades: Optional[List[int]] = None,
    source_path_filters: Optional[List[str]] = None,
) -> List[Dict]:
    model = get_embed_model()
    index = get_faiss_index()
    store = get_store_data()

    q_emb = embed_query(model, question)
    effective_pre_k = max(pre_k, top_k * 4) if question_wants_list(question) else pre_k
    scores, ids = index.search(q_emb, effective_pre_k)
    requested_chapter = extract_question_chapter_number(question)

    allow_glossary = question_is_glossary(question)
    allow_instructions = question_wants_steps(question)

    debug_pre_candidates: List[Dict] = []
    debug_filtered_out: List[Dict] = []
    candidates: List[Tuple[float, Dict]] = []
    for s, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        raw_score = float(s)
        c = store[int(idx)]
        debug_item = build_debug_candidate(c, raw_score)
        debug_pre_candidates.append(debug_item)

        if raw_score < RETRIEVAL_MIN_SCORE:
            debug_filtered_out.append({**debug_item, "alasan": "raw_score < threshold"})
            continue

        if (not allow_glossary) and is_glossary_chunk(c):
            debug_filtered_out.append({**debug_item, "alasan": "glossary tidak diizinkan"})
            continue

        if (not allow_instructions) and is_instruction_chunk(c):
            debug_filtered_out.append({**debug_item, "alasan": "chunk instruksi tidak diizinkan"})
            continue

        if not is_grade_allowed(c, allowed_grades):
            debug_filtered_out.append({**debug_item, "alasan": "grade buku tidak sesuai"})
            continue

        if not is_chapter_allowed(c, requested_chapter):
            debug_filtered_out.append({**debug_item, "alasan": "chapter tidak sesuai"})
            continue

        if not is_source_allowed(c, source_path_filters):
            debug_filtered_out.append({**debug_item, "alasan": "source path tidak sesuai"})
            continue

        c = dict(c)
        c["cue_patterns"] = dedup_list(c.get("cue_patterns", []))
        c["categories"] = dedup_list(c.get("categories", []))

        final = raw_score + score_adjust(question, c)
        c["_raw_score"] = raw_score
        c["_final_score"] = final
        candidates.append((final, c))

    candidates.sort(key=lambda x: x[0], reverse=True)

    initial = [c for _, c in candidates[:max(8, top_k)]]

    expanded = list(initial)
    seen = set()
    final_chunks = []
    for c in expanded:
        cid = c.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        final_chunks.append(c)

    if question_wants_list(question):
        preferred = [c for c in final_chunks if prefer_1_to_5(c)]
        others = [c for c in final_chunks if not prefer_1_to_5(c)]
        final_chunks = preferred + others
    selected_chunks = final_chunks[:top_k]
    debug_kept_candidates = [
        build_debug_candidate(c, c.get("_raw_score", 0.0), c.get("_final_score", 0.0))
        for _, c in candidates
    ]

    if RETRIEVAL_DEBUG:
        debug_print_retrieval(
            question=question,
            pre_k=effective_pre_k,
            top_k=top_k,
            threshold=RETRIEVAL_MIN_SCORE,
            pre_candidates=debug_pre_candidates,
            filtered_out=debug_filtered_out,
            kept_candidates=debug_kept_candidates,
            final_chunks=selected_chunks,
            allowed_grades=allowed_grades,
            requested_chapter=requested_chapter,
            allow_glossary=allow_glossary,
            allow_instructions=allow_instructions,
        )

    return selected_chunks


def pretty_print(chunks: List[Dict], max_chars: int = 240):
    print("\n===== HASIL RETRIEVAL (FINAL) =====\n")
    for i, c in enumerate(chunks, start=1):
        heading = extract_first_heading(c) or "Tanpa heading"
        cues = c.get("cue_patterns", [])
        cats = c.get("categories", [])
        block_ids = ", ".join(c.get("block_ids", [])) or "-"
        print(f"[{i}] {c.get('chunk_id')} | {c.get('source_book')} | Heading: {heading}")
        print(f"    Block: {block_ids} | Cues: {cues} | Categories: {cats}")
        text = (c.get("content") or "").strip().replace("\n", " ")
        print(f"    {text[:max_chars]}{'...' if len(text) > max_chars else ''}\n")

if __name__ == "__main__":
    q = input("Masukkan pertanyaan: ").strip()
    chunks = retrieve(q, top_k=TOP_K, pre_k=PRE_K)
    pretty_print(chunks)
