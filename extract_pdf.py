import re #Regex
import pdfplumber
from adaptive_chunking import Block
from section_heading_config import is_section_heading

NOISE_LINE_PATTERNS = [
    r"^Sumber:\s*.*$",
    r"^Gambar\s*\d+(\.\d+)?\s+.*$",
    r"^\d+\s+Ilmu Pengetahuan.*$",
    r"^\d+\s*$",  
]

FRONT_MATTER_PATTERNS = [
    r"^https?://.*$",
    r"^isbn\b.*$",
    r"^editor\b.*$",
    r"^editor visual\b.*$",
    r"^ilustrator\b.*$",
    r"^penulis\b.*$",
    r"^penelaah\b.*$",
    r"^penyelia\b.*$",
    r"^prakata\b.*$",
    r"^kata pengantar\b.*$",
    r"^daftar isi\b.*$",
    r"^profil pelaku perbukuan\b.*$",
    r"^glosarium\b.*$",
    r"^tentang buku ini\b.*$",
    r"^ada apa di buku ini.*$",
]


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(re.match(p, s, flags=re.IGNORECASE) for p in NOISE_LINE_PATTERNS)


def is_front_matter_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "..." in s or "................................................................" in s:
        return True
    if re.match(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", s, flags=re.IGNORECASE):
        return True
    return any(re.match(p, s, flags=re.IGNORECASE) for p in FRONT_MATTER_PATTERNS)


# --- heading detection (hierarchical and configurable) ---
CONTENT_HEADING_PATTERNS = [
    r"^Bab\s+\d+",
    r"^Topik\s+[A-Z]\s*:",
    r"^[A-Z]\.\s+\S+",
]


def classify_heading_line(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None

    if is_section_heading(s):
        return "section_heading"

    if any(re.match(p, s, flags=re.IGNORECASE) for p in CONTENT_HEADING_PATTERNS):
        return "content_heading"

    if 1 < len(s.split()) <= 6 and not re.search(r"[.!?]$", s):
        lower = s.lower()
        if not re.search(r"\b(adalah|merupakan|yaitu|disebut|karena|sehingga|dengan|setelah|ketika|jika|untuk)\b", lower):
            if not lower.startswith(("dan ", "atau ", "karena ", "sehingga ", "sekarang,", "pada ", "dalam ", "pernahkah ", "tahukah ", "yuk ", "mari ")):
                if s[0].isupper():
                    return "content_heading"

    if len(s.split()) == 1:
        return None

    return None

STEP_PAT = re.compile(r"^\s*(\d+)\.\s+.+$")
ALPHA_STEP_PAT = re.compile(r"^\s*([a-z])\.\s+.+$")

def extract_blocks(pdf_path: str) -> list[Block]:
    blocks: list[Block] = []
    bid = 1

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [ln.rstrip() for ln in text.split("\n")]

            buffer: list[str] = []

            def flush_paragraph():
                nonlocal bid, buffer
                if buffer:
                    para = " ".join(x.strip() for x in buffer if x.strip())
                    if len(para) >= 20:
                        blocks.append(Block(id=f"P{bid}", text=para, block_type="paragraph"))
                        bid += 1
                    buffer = []

            for ln in lines:
                ln = ln.strip()

                if is_noise_line(ln):
                    continue

                if is_front_matter_line(ln):
                    continue

                # empty line -> paragraph boundary
                if ln == "":
                    flush_paragraph()
                    continue

                # heading line -> its own block
                heading_role = classify_heading_line(ln)
                if heading_role:
                    flush_paragraph()
                    blocks.append(
                        Block(
                            id=f"H{bid}",
                            text=ln,
                            block_type="heading",
                            meta={"heading_role": heading_role},
                        )
                    )
                    bid += 1
                    continue

                # numbered/alphabetic step -> split as its own paragraph block
                if STEP_PAT.match(ln) or ALPHA_STEP_PAT.match(ln):
                    flush_paragraph()
                    blocks.append(Block(id=f"S{bid}", text=ln, block_type="paragraph"))
                    bid += 1
                    continue

                buffer.append(ln)

            flush_paragraph()

    return blocks


if __name__ == "__main__":
    bs = extract_blocks("data/ipas.pdf")
    print("blocks:", len(bs))
    for b in bs[:30]:
        print(b.id, b.block_type, b.text[:80])
