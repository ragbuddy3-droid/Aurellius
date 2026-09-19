import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SECTION_HEADINGS = [
    "Pertanyan Esensial",
    "Pertanyaan Esensial",
    "Pertanyaan Kunci",
    "Pendahuluan",
    "Siap-Siap Belajar",
    "Tujuan Pembelajaran",
    "Kata Kunci",
    "Peta Konsep",
    "Peta Materi",
    "Cover Bab",
    "Ayo, Mengamati",
    "Ayo Mengamati",
    "Ayo, Mengamati!",
    "Ayo Mengamati!",
    "Ayo, Menyelidiki",
    "Ayo Menyelidiki",
    "Ayo, Menyimpulkan",
    "Ayo Menyimpulkan",
    "Ayo, Membaca",
    "Ayo Membaca",
    "Ayo, Memahami",
    "Ayo Memahami",
    "Ayo, Mencoba",
    "Ayo Mencoba",
    "Ayo, Mencoba!",
    "Ayo Mencoba!",
    "Ayo, Diskusi",
    "Ayo Diskusi",
    "Ayo, Berdiskusi",
    "Ayo Berdiskusi",
    "Ayo, Berdiskusi!",
    "Ayo Berdiskusi!",
    "Ayo, Bercerita",
    "Ayo Bercerita",
    "Ayo, Bernyanyi",
    "Ayo Bernyanyi",
    "Ayo, Bermain",
    "Ayo Bermain",
    "Ayo, Bermain Peran",
    "Ayo Bermain Peran",
    "Ayo, Berkreasi",
    "Ayo Berkreasi",
    "Ayo, Berlatih",
    "Ayo Berlatih",
    "Ayo, Menulis",
    "Ayo Menulis",
    "Ayo, Menyimak",
    "Ayo Menyimak",
    "Ayo, Wawancara",
    "Ayo Wawancara",
    "Ayo, Sampaikan!",
    "Ayo Sampaikan!",
    "Ayo, Mencari Tahu",
    "Ayo Mencari Tahu",
    "Ayo, Bertanya",
    "Ayo Bertanya",
    "Ayo, Berpendapat",
    "Ayo Berpendapat",
    "Ayo, Menggambar",
    "Ayo Menggambar",
    "Ayo, Kampanye",
    "Ayo Kampanye",
    "Ayo, Tampilkan",
    "Ayo Tampilkan",
    "Ayo, Berkarya",
    "Ayo Berkarya",
    "Ayo, Berekspresi",
    "Ayo Berekspresi",
    "Ayo, Menemukan",
    "Ayo Menemukan",
    "Ayo, Menjodohkan",
    "Ayo Menjodohkan",
    "Ayo, Mengingat Kembali",
    "Ayo Mengingat Kembali",
    "Ayo, Merenungkan",
    "Ayo Merenungkan",
    "Ayo, Kamu Bisa",
    "Ayo Kamu Bisa",
    "Lakukan Bersama",
    "Mari Mencoba",
    "Tugas",
    "Mari Refleksikan",
    "Refleksi",
    "Releksi",
    "Belajar Lebih Lanjut",
    "Lihat di Lingkungan Sekitar",
    "Lihat di Lingkungan Sekitarmu",
    "Memilih Tantangan",
    "Proyek Belajar",
    "Praktik Berpancasila",
    "Uji Kompetensi",
    "Uji Pemahaman",
    "Apa yang Sudah Aku Pelajari?",
    "Apa yang Sudah Aku Pelajari",
    "Kosakata Baru",
    "Pengayaan",
    "Pembelajaran Alternatif",
    "Jelajah Nusantara",
]


DEFAULT_SECTION_MODES: Dict[str, str] = {
    "ayo mengamati": "INSTRUKSI",
    "ayo menyelidiki": "INSTRUKSI",
    "ayo menyimpulkan": "INSTRUKSI",
    "ayo membaca": "GENERAL",
    "ayo memahami": "GENERAL",
    "ayo mencoba": "INSTRUKSI",
    "mari mencoba": "INSTRUKSI",
    "ayo diskusi": "INSTRUKSI",
    "ayo berdiskusi": "INSTRUKSI",
    "ayo mencari tahu": "INSTRUKSI",
    "ayo menggambar": "INSTRUKSI",
    "ayo kampanye": "INSTRUKSI",
    "ayo tampilkan": "INSTRUKSI",
    "ayo berkarya": "INSTRUKSI",
    "ayo mengingat kembali": "EVALUASI",
    "ayo merenungkan": "EVALUASI",
    "ayo kamu bisa": "EVALUASI",
    "lakukan bersama": "INSTRUKSI",
    "tugas": "INSTRUKSI",
    "mari refleksikan": "EVALUASI",
    "refleksi": "EVALUASI",
    "releksi": "EVALUASI",
    "uji kompetensi": "EVALUASI",
    "uji pemahaman": "EVALUASI",
    "apa yang sudah aku pelajari": "EVALUASI",
}


TEACHER_METADATA_DIR = Path("uploads") / "teacher_books" / "metadata"


def normalize_heading_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def strip_activity_prefix(text: str) -> str:
    return re.sub(
        r"^\s*aktivitas\s+\d+(\.\d+)?\s*",
        "",
        text or "",
        flags=re.IGNORECASE,
    ).strip()


def load_teacher_heading_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not TEACHER_METADATA_DIR.exists():
        return entries

    for metadata_path in sorted(TEACHER_METADATA_DIR.glob("*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for item in payload.get("heading_entries", []):
            name = (item.get("name") or "").strip()
            section_mode = (item.get("section_mode") or "GENERAL").strip().upper()
            if not name:
                continue
            if section_mode not in {"GENERAL", "INSTRUKSI", "EVALUASI"}:
                section_mode = "GENERAL"
            entries.append(
                {
                    "name": name,
                    "section_mode": section_mode,
                }
            )

    return entries


def get_all_section_headings() -> List[str]:
    merged = list(DEFAULT_SECTION_HEADINGS)
    seen = {normalize_heading_text(item) for item in merged}

    for entry in load_teacher_heading_entries():
        normalized = normalize_heading_text(entry["name"])
        if normalized and normalized not in seen:
            merged.append(entry["name"])
            seen.add(normalized)

    return merged


def get_section_mode_map() -> Dict[str, str]:
    mode_map = dict(DEFAULT_SECTION_MODES)
    for entry in load_teacher_heading_entries():
        mode_map[normalize_heading_text(entry["name"])] = entry["section_mode"]
    return mode_map


SECTION_HEADING_SET = {normalize_heading_text(item) for item in get_all_section_headings()}


def is_section_heading(text: str) -> bool:
    raw = strip_activity_prefix(text)
    normalized = normalize_heading_text(raw)
    if not normalized:
        return False

    if normalized in SECTION_HEADING_SET:
        return True

    return any(
        normalized.startswith(item + " ")
        for item in SECTION_HEADING_SET
        if len(item.split()) >= 2
    )


def get_section_mode(text: str) -> str:
    normalized = normalize_heading_text(strip_activity_prefix(text))
    return get_section_mode_map().get(normalized, "GENERAL")


def build_section_split_regex(section_headings: Iterable[str]) -> re.Pattern:
    escaped = sorted(
        {re.escape(item.strip()) for item in section_headings if item and item.strip()},
        key=len,
        reverse=True,
    )
    prefixed = r"(?:Aktivitas\s+\d+(?:\.\d+)?\s+)?"
    combined = "|".join(prefixed + item for item in escaped)
    return re.compile(f"({combined})", flags=re.IGNORECASE)
