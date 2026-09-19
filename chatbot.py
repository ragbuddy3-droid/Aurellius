import os
import time
from typing import Dict, List

import torch
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DEFAULT_CHAT_PROVIDER,
    GEMINI_MODEL_NAME,
    PRE_K,
    QWEN_MODEL_NAME,
    QWEN_MAX_NEW_TOKENS,
    QWEN_TEMPERATURE,
    TOP_K,
)
from retrieval import retrieve


SUPPORTED_PROVIDERS = {"gemini", "qwen"}
_QWEN_TOKENIZER = None
_QWEN_MODEL = None


def normalize_provider(provider: str | None) -> str:
    value = (provider or DEFAULT_CHAT_PROVIDER or "gemini").strip().lower()
    return value if value in SUPPORTED_PROVIDERS else "gemini"


def question_wants_list(question: str) -> bool:
    q = (question or "").lower()
    return any(
        key in q
        for key in [
            "apa saja",
            "sifat-sifat",
            "sifat sifat",
            "jenis-jenis",
            "jenis jenis",
            "macam-macam",
            "macam macam",
            "daftar",
        ]
    )


def question_asks_definition(question: str) -> bool:
    q = (question or "").lower().strip()
    return (
        q.startswith("apa itu ")
        or q.startswith("apakah itu ")
        or q.endswith(" adalah")
        or "pengertian" in q
        or "definisi" in q
    )


def question_asks_why(question: str) -> bool:
    q = (question or "").lower().strip()
    return any(key in q for key in ["mengapa", "kenapa", "bagaimana bisa"])


def resolve_top_k(question: str, top_k: int) -> int:
    if question_wants_list(question):
        return max(top_k, 6)
    if question_asks_why(question):
        return max(top_k, 4)
    return top_k


def clean_answer_text(question: str, answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    text = text.replace("\r\n", "\n")

    if question_wants_list(question):
        text = text.replace(" * ", "\n- ")
        if text.startswith("* "):
            text = "- " + text[2:]
        if ":\n- " not in text and ": - " in text:
            text = text.replace(": - ", ":\n- ")

    return text.strip()


def format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        headings = chunk.get("headings") or ["-"]
        categories = ", ".join(chunk.get("categories", [])) or "-"
        source_book = chunk.get("source_book", "-")
        content = (chunk.get("content") or "").strip()
        parts.append(
            "\n".join(
                [
                    f"[Konteks {i}]",
                    f"Sumber buku: {source_book}",
                    f"Heading: {headings[0]}",
                    f"Kategori: {categories}",
                    "Isi:",
                    content,
                ]
            )
        )
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: List[Dict], provider: str | None = None) -> str:
    context = format_context(chunks)
    resolved_provider = normalize_provider(provider)
    wants_list = question_wants_list(question)
    asks_definition = question_asks_definition(question)
    asks_why = question_asks_why(question)
    task_block = (
        "- Jawab dalam bentuk daftar poin.\n"
        "- Sebutkan setiap poin penting yang ditemukan pada konteks.\n"
        "- Jika konteks memuat beberapa sifat, jenis, langkah, atau contoh, tampilkan semuanya yang relevan sebagai poin terpisah.\n"
        "- Jangan membuka jawaban dengan sapaan seperti 'Halo'.\n"
        "- Jangan berkata bahwa materi tidak lengkap kecuali konteks benar-benar tidak cukup untuk menjawab inti pertanyaan."
        if wants_list
        else
        "- Jawab sebab atau alasan secara langsung dalam 2-4 kalimat sederhana.\n"
        "- Jika konteks tidak memakai kata yang persis sama dengan pertanyaan, gunakan penjelasan terdekat yang masih relevan.\n"
        "- Gabungkan petunjuk dari beberapa konteks jika isinya saling mendukung.\n"
        "- Jangan membuka jawaban dengan sapaan seperti 'Halo'.\n"
        "- Jangan terlalu cepat mengatakan jawaban tidak ada jika konteks masih menjelaskan penyebab atau proses yang terkait."
        if asks_why
        else
        "- Mulai jawaban dengan definisi atau inti jawaban secara langsung.\n"
        "- Jika konteks tidak memberi definisi persis, rangkum inti penjelasan terdekat menjadi 2-4 kalimat yang tetap jelas.\n"
        "- Setelah definisi, boleh tambahkan satu atau dua sifat atau contoh yang relevan.\n"
        "- Jangan membuka jawaban dengan sapaan seperti 'Halo'.\n"
        "- Jangan terlalu cepat mengatakan materi tidak ada jika konteks masih memberi petunjuk yang cukup."
        if asks_definition
        else
        "- Berikan jawaban yang singkat tetapi jelas.\n"
        "- Utamakan penjelasan yang mudah dipahami anak SD.\n"
        "- Jangan membuka jawaban dengan sapaan seperti 'Halo'.\n"
        "- Jika cocok, gunakan poin-poin sederhana."
    )
    provider_rules = (
        "\nAturan tambahan khusus:\n"
        "- Gunakan hanya informasi yang tertulis di konteks.\n"
        "- Jangan menebak, jangan mengarang, dan jangan menambahkan pengetahuan umum di luar konteks.\n"
        "- Jika konteks tidak cukup, jawab persis: Maaf, jawaban belum ditemukan pada materi yang tersedia.\n"
        "- Jangan memberi contoh tambahan kecuali contoh itu memang ada di konteks.\n"
        "- Jawaban maksimal 3 kalimat atau 4 poin singkat.\n"
        if resolved_provider == "qwen"
        else ""
    )
    return f"""Kamu adalah chatbot pendidikan untuk siswa Sekolah Dasar.

Jawablah pertanyaan hanya berdasarkan konteks yang diberikan.
Gunakan bahasa Indonesia yang sederhana, jelas, dan ramah untuk anak SD.
Jika konteks benar-benar tidak cukup untuk menjawab inti pertanyaan, katakan dengan jujur bahwa jawaban belum ditemukan dari materi yang tersedia.
Namun, jika konteks masih memberi petunjuk yang cukup, buatlah jawaban yang membantu tanpa menyebutkan keterbatasan konteks.
Jangan menambahkan informasi yang tidak didukung konteks.

Pertanyaan:
{question}

Konteks:
{context}

Tugas:
{task_block}
{provider_rules}
"""


def call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY belum diset.")

    client = genai.Client(api_key=api_key)
    last_error = None

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
            return response.text or ""
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            is_retryable = any(
                key in message
                for key in [
                    "503",
                    "unavailable",
                    "high demand",
                    "rate limit",
                    "429",
                    "deadline exceeded",
                    "timeout",
                ]
            )

            if attempt == 3 or not is_retryable:
                raise

            wait_seconds = 3 * (attempt + 1)
            print(
                f"[Gemini retry] percobaan {attempt + 1} gagal: {exc}. "
                f"Mencoba lagi dalam {wait_seconds} detik..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Gagal memanggil Gemini setelah beberapa percobaan: {last_error}")


def get_qwen_local_components():
    global _QWEN_MODEL, _QWEN_TOKENIZER

    if _QWEN_MODEL is not None and _QWEN_TOKENIZER is not None:
        return _QWEN_TOKENIZER, _QWEN_MODEL

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    _QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    _QWEN_MODEL.to(device)
    _QWEN_MODEL.eval()
    return _QWEN_TOKENIZER, _QWEN_MODEL


def call_qwen(prompt: str) -> str:
    try:
        tokenizer, model = get_qwen_local_components()
    except Exception as exc:
        raise RuntimeError(
            f"Model Qwen lokal gagal dimuat dari HuggingFace: {QWEN_MODEL_NAME}. "
            f"Pastikan modelnya tersedia dan dependency transformers sudah terpasang. Detail: {exc}"
        ) from exc

    messages = [
        {"role": "system", "content": "Kamu adalah chatbot pendidikan untuk siswa SD."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = (
            "System: Kamu adalah chatbot pendidikan untuk siswa SD.\n\n"
            f"User: {prompt}\n\nAssistant:"
        )

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=QWEN_TEMPERATURE > 0,
            temperature=QWEN_TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def call_model(provider: str, prompt: str) -> str:
    resolved = normalize_provider(provider)
    if resolved == "qwen":
        return call_qwen(prompt)
    return call_gemini(prompt)


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    pre_k: int = PRE_K,
    allowed_grades: List[int] | None = None,
    provider: str | None = None,
) -> tuple[str, List[Dict]]:
    effective_top_k = resolve_top_k(question, top_k)
    chunks = retrieve(question, top_k=effective_top_k, pre_k=pre_k, allowed_grades=allowed_grades)
    if not chunks:
        return "Maaf, aku belum menemukan materi yang cukup sesuai di buku untuk menjawab pertanyaan itu.", []

    prompt = build_prompt(question, chunks, provider or DEFAULT_CHAT_PROVIDER)
    answer_text = call_model(provider or DEFAULT_CHAT_PROVIDER, prompt)
    answer = clean_answer_text(question, answer_text)
    return answer, chunks


def print_sources(chunks: List[Dict]) -> None:
    print("\nSumber konteks:")
    for i, chunk in enumerate(chunks, start=1):
        heading = (chunk.get("headings") or ["Tanpa heading"])[0]
        categories = ", ".join(chunk.get("categories", [])) or "-"
        print(
            f"{i}. {chunk.get('chunk_id')} | {chunk.get('source_book')} | "
            f"{categories} | {heading}"
        )


if __name__ == "__main__":
    print("Chatbot Buku Pelajaran SD Berbasis RAG")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        question = input("Pertanyaan: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer, chunks = generate_answer(question)
            print("\nJawaban:")
            print(answer)
            if chunks:
                print_sources(chunks)
            print()
        except Exception as exc:
            print(f"\nTerjadi kesalahan: {exc}\n")
