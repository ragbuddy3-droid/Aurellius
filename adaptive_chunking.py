from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from section_heading_config import (
    build_section_split_regex,
    get_all_section_headings,
    get_section_mode,
    is_section_heading,
)

#Struktur Data
@dataclass
class Block:
    id: str
    text: str
    block_type: str = "paragraph"
    meta: Optional[Dict] = None


@dataclass
class LabeledBlock:
    id: str
    text: str
    block_type: str
    cue_pattern: str
    category: str
    meta: Optional[Dict] = None


@dataclass
class Chunk:
    chunk_id: str
    texts: List[str]
    block_ids: List[str]
    cue_patterns: List[str]
    categories: List[str]
    meta: Dict

    @property
    def content(self) -> str:
        return "\n".join(self.texts)


#Discourse Marker
CP_HEADING = "NEW_SUBTOPIC_HEADING"
CP_SECTION = "SECTION_HEADING"
CP_IMPERATIVE = "IMPERATIVE_TASK"
CP_EVALUATIVE = "EVALUATIVE_QUESTION"
CP_DEFINITION = "DEFINITION"
CP_CAUSE_EFFECT = "CAUSE_EFFECT"
CP_EXAMPLE = "EXAMPLE_ILLUSTRATION"
CP_NARRATIVE = "NARRATIVE_SEQUENCE"
CP_FACT = "FACT_EXPLANATION"
CP_INTRO = "PROMPT_INTRO"
CP_GLOSSARY = "GLOSSARY_BOX"
CP_META = "BOOK_META"


def normalize_text(t: str) -> str:
    t = (t or "").replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


IMPERATIVE_VERBS = {
    "amati", "perhatikan", "lakukan", "siapkan", "diskusikan",
    "tuliskan", "jawablah", "sebutkan", "urutkan", "pasangkan",
    "buatlah", "bacalah", "cermati", "pahami", "gunakan",
    "kerjakan", "cobalah", "tentukan", "prediksi", "isilah",
    "lengkapilah", "tebaklah", "ukur", "gambar", "hitung",
    "jelaskan", "uraikan", "pilihlah", "lingkarilah",
}

EVAL_TRIGGERS = {
    "latihan", "soal", "ujian",
    "jawablah", "isilah", "pasangkan", "urutkan", "pilihlah",
    "lengkapilah", "teka-teki silang", "diagram venn",
    "refleksikan", "refleksi", "mari refleksikan", "mari refleksi",
    "evaluasi", "penilaian", "apa yang sudah aku pelajari"
}

#Pembuka topik
PROMPT_WORDS = {
    "tahukah", "yuk", "mari", "pernahkah", "selamat belajar", "masih ingat"
}

ACTIVITY_HEADINGS = {
    "lakukan bersama",
    "mari mencoba",
    "ayo,",
    "ayo ",
    "belajar lebih lanjut",
    "projek", "proyek",
}

GLOSSARY_HEADINGS = {"kosakata baru"}

QUESTION_WORDS = {
    "apa", "mengapa", "bagaimana", "kapan", "siapa",
    "di mana", "dimana", "kenapa",
}

STEP_PREFIXES = (
    "saat kalian",
    "ketika kalian",
    "jika kalian",
    "bila kalian",
    "setelah itu",
    "kemudian",
    "lalu",
)

CONTENT_CUES = {CP_DEFINITION, CP_CAUSE_EFFECT, CP_EXAMPLE, CP_NARRATIVE, CP_FACT, CP_INTRO}


def looks_like_book_meta(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False

    patterns = [
        r"\bisbn\b",
        r"\bkementerian\b",
        r"\brepublik indonesia\b",
        r"\bkemendikbud\b",
        r"\bpenulis\b",
        r"\bjil\.\b",
        r"\bhak cipta\b",
        r"\b(pusat|balai) kurikulum\b",
    ]
    if any(re.search(p, tl) for p in patterns):
        return True

    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) >= 30:
        upper_ratio = sum(ch.isupper() for ch in letters) / max(1, len(letters))
        if upper_ratio > 0.85:
            return True

    return False


def is_numbered_line(t: str) -> bool:
    return bool(re.match(r"^\s*\d+\.\s+", t))


def strip_number_prefix(t: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", t).strip()


def is_alpha_list_line(t: str) -> bool:
    return bool(re.match(r"^\s*[a-z]\.\s+", t))


def strip_alpha_prefix(t: str) -> str:
    return re.sub(r"^\s*[a-z]\.\s*", "", t).strip()


def looks_like_numbered_question(t: str) -> bool:
    t_norm = normalize_text(t)
    if not is_numbered_line(t_norm):
        return False
    after = strip_number_prefix(t_norm).lower()

    if "?" in t_norm:
        return True

    for qw in QUESTION_WORDS:
        if after.startswith(qw + " ") or after == qw:
            return True

    return False


def starts_with_imperative(t: str) -> bool:
    t = normalize_text(t).lower()
    if not t:
        return False
    first = re.split(r"\s+", t, maxsplit=1)[0]
    first = re.sub(r"[^\w-]", "", first)

    if first in IMPERATIVE_VERBS:
        return True

    if first.endswith("lah") and first[:-3] in IMPERATIVE_VERBS:
        return True

    if first.startswith(("meng", "meny", "men", "mem", "me")):
        return True

    return False


def contains_imperative_anywhere(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False
    words = re.findall(r"[a-zA-Z-]+", tl)
    for w in words:
        w = re.sub(r"[^\w-]", "", w)
        if w in IMPERATIVE_VERBS:
            return True
        if w.endswith("lah") and w[:-3] in IMPERATIVE_VERBS:
            return True
    return False


def numbered_line_is_heading(t: str) -> bool:
    t_norm = normalize_text(t)
    if not is_numbered_line(t_norm):
        return False

    if looks_like_numbered_question(t_norm):
        return False

    after = strip_number_prefix(t_norm).lower()
    toks = after.split()

    for tok in toks[:3]:
        tok_clean = re.sub(r"[^\w-]", "", tok)
        if tok_clean in IMPERATIVE_VERBS:
            return False
        if tok_clean.endswith("lah") and tok_clean[:-3] in IMPERATIVE_VERBS:
            return False

    for pref in STEP_PREFIXES:
        if after.startswith(pref):
            return False

    return len(toks) <= 6


def is_heading_text(t: str) -> bool:
    t = normalize_text(t)
    tl = t.lower()
    if not t:
        return False

    if is_section_heading(t):
        return True

    if re.match(r"^\s*(bab|topik)\s+\w+", tl):
        return True

    if re.match(r"^\s*[A-Z]\.\s+\S+", t):
        return True

    if any(tl.startswith(h) for h in GLOSSARY_HEADINGS):
        return True

    if any(tl.startswith(h) for h in ACTIVITY_HEADINGS):
        return True

    if is_numbered_line(t):
        return numbered_line_is_heading(t)

    if len(t.split()) == 1:
        return False

    short = 1 < len(t.split()) <= 6 and not re.search(r"[.!?]$", t)
    has_definition_word = bool(re.search(r"\b(adalah|merupakan|yaitu|disebut|karena|sehingga|dengan|setelah|ketika|jika|untuk)\b", tl))
    has_question = "?" in t
    starts_like_sentence = tl.startswith(("dan ", "atau ", "karena ", "sehingga ", "sekarang,", "pada ", "dalam "))
    if short and not has_definition_word and not has_question and not starts_like_sentence:
        return True

    return False


def looks_like_instruction_intro(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False
    patterns = [
        r"\blakukan langkah[- ]langkah\b",
        r"\bikuti langkah[- ]langkah\b",
        r"\bpetunjuk di bawah ini\b",
        r"\bkegiatan berikut\b",
        r"\blangkah berikut\b",
        r"\bsebelum melakukan\b",
    ]
    return any(re.search(p, tl) for p in patterns)


def looks_like_evaluative(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False
    if looks_like_numbered_question(t):
        return True
    return any(trg in tl for trg in EVAL_TRIGGERS)


def looks_like_continuation(prev_text: str, cur_text: str) -> bool:
    prev = normalize_text(prev_text)
    cur = normalize_text(cur_text)
    if not prev or not cur:
        return False

    if re.search(r"[.!?…:]$", prev):
        return False

    cur_l = cur.lstrip()
    starts_lower = bool(re.match(r"^[a-z]", cur_l))
    starts_connector = cur_l.lower().startswith((
        "dan ", "atau ", "serta ", "yang ", "hasil ", "kemudian", "lalu", "setelah", "karena", "sehingga"
    ))

    shortish = len(cur.split()) <= 25 
    return (starts_lower or starts_connector) and shortish


def section_mode(section_heading: Optional[str]) -> Optional[str]:
    if not section_heading:
        return None
    mode = get_section_mode(section_heading)
    if mode == "GENERAL":
        return None
    return mode


def numbered_line_is_title_like(t: str) -> bool:
    t_norm = normalize_text(t)
    if not is_numbered_line(t_norm):
        return False

    after = strip_number_prefix(t_norm)
    lower = after.lower()
    words = after.split()
    if not (2 <= len(words) <= 5):
        return False
    if re.search(r"[!?]$", after):
        return False
    if starts_with_imperative(after):
        return False
    if re.search(r"\b(adalah|merupakan|yaitu|disebut|karena|sehingga|ketika|setelah|sebelum|jika|untuk|dengan)\b", lower):
        return False
    if lower.startswith(("dan ", "atau ", "karena ", "sehingga ", "sekarang,", "pada ", "dalam ")):
        return False
    return after[:1].isupper()


ROMAN_MAP = {
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


def parse_chapter_heading(text: str) -> tuple[Optional[str], Optional[int]]:
    t = normalize_text(text)
    m = re.search(r"\bBab\s+([0-9IVX]+)\b", t, flags=re.IGNORECASE)
    if not m:
        return None, None

    raw = m.group(1).upper()
    if raw.isdigit():
        num = int(raw)
    else:
        num = ROMAN_MAP.get(raw)

    if num is None:
        return None, None

    return f"Bab {num}", num




SECTION_SPLIT_RE = build_section_split_regex(get_all_section_headings())


def split_blocks_on_sections(blocks: List[Block]) -> List[Block]:
    out: List[Block] = []
    for b in blocks:
        text = normalize_text(b.text)
        if not text:
            continue

        parts = SECTION_SPLIT_RE.split(text)
        if len(parts) == 1:
            out.append(b)
            continue

        buf = ""
        idx = 0
        piece_no = 1

        while idx < len(parts):
            chunk_raw = parts[idx]
            if chunk_raw is None:
                idx += 1
                continue

            chunk = chunk_raw.strip()
            if not chunk:
                idx += 1
                continue

            if is_section_heading(chunk):
                if buf.strip():
                    out.append(Block(
                        id=f"{b.id}_a{piece_no}",
                        text=buf.strip(),
                        block_type=b.block_type,
                        meta=b.meta,
                    ))
                    piece_no += 1
                    buf = ""

                out.append(Block(
                    id=f"{b.id}_s{piece_no}",
                    text=chunk.strip(),
                    block_type="heading",
                    meta={"heading_role": "section_heading"},
                ))
                piece_no += 1
            else:
                if buf:
                    buf += " " + chunk
                else:
                    buf = chunk

            idx += 1

        if buf.strip():
            out.append(Block(
                id=f"{b.id}_a{piece_no}",
                text=buf.strip(),
                block_type=b.block_type,
                meta=b.meta,
            ))

    return out


def detect_cue_pattern(block: Block) -> str:
    t = normalize_text(block.text)
    tl = t.lower()

    tl_norm = re.sub(r"\s+", " ", tl)

    heading_role = (block.meta or {}).get("heading_role")
    if heading_role == "section_heading":
        if "kosakata baru" in tl:
            return CP_GLOSSARY
        return CP_SECTION

    if heading_role == "content_heading":
        return CP_HEADING

    if looks_like_evaluative(t) and len(tl_norm.split()) <= 10:
        return CP_EVALUATIVE

    if "kosakata baru" in tl:
        return CP_GLOSSARY
    
    if re.search(r"(^|\n)\s*[^:\n]{2,40}\s*:\s+\S+", t):
        if len(t) <= 400:
            return CP_GLOSSARY

    if looks_like_book_meta(t):
        return CP_META

    if is_heading_text(t):
        if any(tl.startswith(h) for h in GLOSSARY_HEADINGS):
            return CP_GLOSSARY

        if looks_like_evaluative(t):
            return CP_EVALUATIVE

        if any(tl.startswith(h) for h in ACTIVITY_HEADINGS):
            return CP_IMPERATIVE

        if looks_like_numbered_question(t):
            return CP_EVALUATIVE

        return CP_HEADING
    
    if re.match(r"^\s*\d+\.\s+", t):
        after = re.sub(r"^\s*\d+\.\s+", "", tl_norm)
        first_word = after.split(" ", 1)[0] if after else ""
        if first_word not in IMPERATIVE_VERBS:
            if "?" not in t:
                return CP_EVALUATIVE


    if is_numbered_line(t) and not numbered_line_is_heading(t):
        if looks_like_numbered_question(t):
            return CP_EVALUATIVE
        return CP_IMPERATIVE

    if starts_with_imperative(t) or tl.startswith("ayo,") or tl.startswith("ayo "):
        if looks_like_evaluative(t):
            return CP_EVALUATIVE
        return CP_IMPERATIVE
    
    if looks_like_evaluative(t):
        return CP_EVALUATIVE

    if any(p in tl for p in PROMPT_WORDS):
        return CP_INTRO

    if re.search(r"\b(adalah|merupakan|yaitu|disebut)\b", tl):
        return CP_DEFINITION

    if re.search(r"\b(karena|sehingga|oleh karena itu)\b", tl):
        return CP_CAUSE_EFFECT

    if re.search(r"\b(misalnya|contohnya|seperti)\b", tl):
        return CP_EXAMPLE

    if re.search(r"\b(dahulu|suatu hari|kemudian|tinggallah)\b", tl):
        return CP_NARRATIVE

    return CP_FACT


def cue_to_category(cue: str) -> str:
    if cue == CP_SECTION:
        return "SECTION"
    if cue == CP_IMPERATIVE:
        return "INSTRUKSI"
    if cue == CP_EVALUATIVE:
        return "EVALUASI"
    if cue == CP_NARRATIVE:
        return "NARASI"
    if cue == CP_META:
        return "META"
    if cue == CP_GLOSSARY:
        return "GLOSARIUM"
    return "KONSEP"


def label_blocks(blocks: List[Block]) -> List[LabeledBlock]:
    out: List[LabeledBlock] = []
    for b in blocks:
        cue = detect_cue_pattern(b)
        cat = cue_to_category(cue)
        out.append(
            LabeledBlock(
                id=b.id,
                text=normalize_text(b.text),
                block_type=b.block_type,
                cue_pattern=cue,
                category=cat,
                meta=b.meta or {},
            )
        )
    return out

#Context State
def should_start_new_chunk(prev: Optional[LabeledBlock], cur: LabeledBlock) -> bool:
    if prev is None:
        return True

    if prev.category == "GLOSARIUM" or cur.category == "GLOSARIUM":
        return prev.category != cur.category

    if prev.category == "INSTRUKSI" and cur.category == "INSTRUKSI":
        return False

    if prev.category == "EVALUASI" and cur.category == "EVALUASI":
        return False

    if prev.category != cur.category:
        return True

    return False


#
def build_chunks(labeled: List[LabeledBlock], skip_meta_blocks: bool = True) -> List[Chunk]:
    chunks: List[Chunk] = []

    current_activity: Optional[str] = None  
    active_chapter: Optional[str] = None
    active_chapter_number: Optional[int] = None
    active_section_heading: Optional[str] = None
    active_content_heading: Optional[str] = None

    current_texts: List[str] = []
    current_ids: List[str] = []
    current_cues: List[str] = []
    current_cats: List[str] = []
    current_meta: Dict = {"headings": [], "section_heading": None, "content_heading": None}

    last_effective: Optional[LabeledBlock] = None
    last_text_in_chunk: str = ""
    chunk_counter = 1

    in_glossary_chunk = False

    def fresh_meta() -> Dict:
        headings = []
        if active_chapter:
            headings.append(active_chapter)
        if active_section_heading:
            headings.append(active_section_heading)
        if active_content_heading:
            headings.append(active_content_heading)
        return {
            "headings": headings,
            "chapter": active_chapter,
            "chapter_number": active_chapter_number,
            "section_heading": active_section_heading,
            "content_heading": active_content_heading,
        }

    def flush():
        nonlocal chunk_counter, current_texts, current_ids, current_cues, current_cats, current_meta
        nonlocal last_effective, last_text_in_chunk, in_glossary_chunk
        if not current_texts:
            in_glossary_chunk = False
            return
        chunks.append(
            Chunk(
                chunk_id=f"CH{chunk_counter:04d}",
                texts=current_texts,
                block_ids=current_ids,
                cue_patterns=current_cues,
                categories=current_cats,
                meta=current_meta,
            )
        )
        chunk_counter += 1
        current_texts, current_ids, current_cues, current_cats = [], [], [], []
        current_meta = fresh_meta()
        last_effective = None
        last_text_in_chunk = ""
        in_glossary_chunk = False

    for lb in labeled:
        if skip_meta_blocks and lb.cue_pattern == CP_META:
            continue

        if in_glossary_chunk and lb.cue_pattern != CP_GLOSSARY:
            flush()

        if lb.cue_pattern == CP_SECTION:
            flush()
            current_activity = section_mode(lb.text)
            active_section_heading = lb.text
            active_content_heading = None
            current_meta = fresh_meta()
            continue

        if is_numbered_line(lb.text):
            sec_mode = section_mode(active_section_heading)
            if looks_like_numbered_question(lb.text):
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_EVALUATIVE,
                    category="EVALUASI",
                    meta=lb.meta,
                )
            elif starts_with_imperative(strip_number_prefix(lb.text)):
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_IMPERATIVE,
                    category="INSTRUKSI",
                    meta=lb.meta,
                )
            elif numbered_line_is_title_like(lb.text):
                if current_texts:
                    flush()
                current_activity = None
                active_content_heading = normalize_text(lb.text)
                current_meta = fresh_meta()
                continue
            elif sec_mode == "EVALUASI":
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_EVALUATIVE,
                    category="EVALUASI",
                    meta=lb.meta,
                )
            elif sec_mode == "INSTRUKSI":
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_IMPERATIVE,
                    category="INSTRUKSI",
                    meta=lb.meta,
                )

        if is_alpha_list_line(lb.text):
            sec_mode = section_mode(active_section_heading)
            alpha_text = strip_alpha_prefix(lb.text)
            if "?" in lb.text or looks_like_evaluative(alpha_text):
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_EVALUATIVE,
                    category="EVALUASI",
                    meta=lb.meta,
                )
            elif starts_with_imperative(alpha_text) or current_activity == "INSTRUKSI" or sec_mode == "INSTRUKSI":
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_IMPERATIVE,
                    category="INSTRUKSI",
                    meta=lb.meta,
                )
            elif sec_mode == "EVALUASI":
                lb = LabeledBlock(
                    id=lb.id,
                    text=lb.text,
                    block_type=lb.block_type,
                    cue_pattern=CP_EVALUATIVE,
                    category="EVALUASI",
                    meta=lb.meta,
                )

        if lb.cue_pattern == CP_HEADING:
            chapter_label, chapter_number = parse_chapter_heading(lb.text)
            if current_texts:
                flush()
            current_activity = None
            if chapter_label:
                active_chapter = chapter_label
                active_chapter_number = chapter_number
                active_content_heading = None
            else:
                active_content_heading = lb.text
            current_meta = fresh_meta()
            continue

        if lb.cue_pattern == CP_GLOSSARY:
            if not in_glossary_chunk:
                if current_texts:
                    flush()
                current_activity = None
                in_glossary_chunk = True

            effective_lb = LabeledBlock(
                id=lb.id,
                text=lb.text,
                block_type=lb.block_type,
                cue_pattern=CP_GLOSSARY,
                category="GLOSARIUM",
                meta=lb.meta,
            )

            current_texts.append(effective_lb.text)
            current_ids.append(effective_lb.id)
            current_cues.append(effective_lb.cue_pattern)
            current_cats.append(effective_lb.category)

            if effective_lb.meta:
                current_meta.setdefault("blocks_meta", []).append({effective_lb.id: effective_lb.meta})

            last_effective = effective_lb
            last_text_in_chunk = effective_lb.text
            continue

        if lb.cue_pattern == CP_IMPERATIVE:
            current_activity = "INSTRUKSI"
        elif lb.cue_pattern == CP_EVALUATIVE:
            current_activity = "EVALUASI"
        elif lb.cue_pattern in CONTENT_CUES:
            current_activity = None

        forced_cue = lb.cue_pattern
        forced_cat = lb.category

        if last_effective is not None and looks_like_continuation(last_text_in_chunk, lb.text):
            if last_effective.category == "INSTRUKSI":
                forced_cue = CP_IMPERATIVE
                forced_cat = "INSTRUKSI"
            elif last_effective.category == "EVALUASI":
                forced_cue = CP_EVALUATIVE
                forced_cat = "EVALUASI"

        if current_activity == "INSTRUKSI":
            if forced_cue in (CP_NARRATIVE, CP_FACT, CP_INTRO) and (
                contains_imperative_anywhere(lb.text) or looks_like_instruction_intro(lb.text)
            ):
                forced_cue = CP_IMPERATIVE
                forced_cat = "INSTRUKSI"

            if forced_cue in (CP_FACT, CP_INTRO) and looks_like_continuation(last_text_in_chunk, lb.text):
                forced_cue = CP_IMPERATIVE
                forced_cat = "INSTRUKSI"

        if current_activity == "EVALUASI":
            if forced_cue in (CP_FACT, CP_INTRO) and ("?" in lb.text or looks_like_evaluative(lb.text)):
                forced_cue = CP_EVALUATIVE
                forced_cat = "EVALUASI"

        effective_category = forced_cat
        if current_activity in ("INSTRUKSI", "EVALUASI"):
            if forced_cue in (CP_IMPERATIVE, CP_EVALUATIVE):
                effective_category = current_activity

        effective_lb = LabeledBlock(
            id=lb.id,
            text=lb.text,
            block_type=lb.block_type,
            cue_pattern=forced_cue,
            category=effective_category,
            meta=lb.meta,
        )

        if should_start_new_chunk(last_effective, effective_lb) and current_texts:
            flush()

        current_texts.append(effective_lb.text)
        current_ids.append(effective_lb.id)
        current_cues.append(effective_lb.cue_pattern)
        current_cats.append(effective_lb.category)

        if effective_lb.meta:
            current_meta.setdefault("blocks_meta", []).append({effective_lb.id: effective_lb.meta})

        last_effective = effective_lb
        last_text_in_chunk = effective_lb.text

    flush()
    return chunks

def print_chunks(chunks: List[Chunk]) -> None:
    for ch in chunks:
        print(ch.chunk_id)
        print("HEADINGS:", ch.meta.get("headings", []))
        print("BLOCK_IDS:", ch.block_ids)
        print("CUE_PATTERNS:", ch.cue_patterns)
        print("CATEGORIES:", ch.categories)
        print("CONTENT:\n", ch.content)
        print("-" * 80)
