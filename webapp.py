import json
import subprocess
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
import traceback
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from sqlalchemy import inspect, text
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

from chatbot import generate_answer
from config import (
    APP_SECRET_KEY,
    DATABASE_URL,
    DEFAULT_CHAT_PROVIDER,
    MAX_UPLOAD_PDF_BYTES,
    MAX_UPLOAD_PDF_MB,
    SQLALCHEMY_TRACK_MODIFICATIONS,
)
from database import db
import models  # noqa: F401
from models import Book, ChatHistory, Heading, ProcessingLog, User


app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_PDF_BYTES + (1024 * 1024)
db.init_app(app)

UPLOAD_DIR = Path("uploads") / "teacher_books"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_METADATA_DIR = UPLOAD_DIR / "metadata"
UPLOAD_METADATA_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent
APP_TIMEZONE = ZoneInfo("Asia/Jakarta")
LEGACY_BOOKS_DIR = PROJECT_ROOT / "data"

ROMAN_GRADE_MAP = {
    "IV": "4",
    "V": "5",
    "VI": "6",
}

LEGACY_SUBJECT_LABELS = {
    "PKN": "Pendidikan Pancasila",
}


def format_file_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} byte"


def get_upload_file_size(file_storage) -> int:
    stream = file_storage.stream
    current_position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(current_position)
    return size


@app.context_processor
def inject_upload_limits() -> Dict[str, Any]:
    return {
        "max_upload_pdf_mb": MAX_UPLOAD_PDF_MB,
        "max_upload_pdf_bytes": MAX_UPLOAD_PDF_BYTES,
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_exc):
    error = (
        f"Ukuran file terlalu besar. Maksimal upload PDF adalah "
        f"{MAX_UPLOAD_PDF_MB} MB per file."
    )
    return render_template(
        "teacher_upload.html",
        title="Upload Buku Guru",
        brand_title="Chatbot/Guru",
        success=None,
        error=error,
        process_result=None,
        form_data={
            "book_title": "",
            "subject": "",
            "grade": "",
            "notes": "",
            "heading_entries": [
                {"name": "", "section_mode": "GENERAL", "description": ""},
            ],
        },
    ), 413


def ensure_database_schema() -> None:
    inspector = inspect(db.engine)
    if not inspector.has_table("users"):
        return

    user_columns = {column["name"] for column in inspector.get_columns("users")}
    if "student_grade" not in user_columns:
        with db.engine.begin() as connection:
            connection.execute(text("ALTER TABLE users ADD COLUMN student_grade INTEGER"))


with app.app_context():
    ensure_database_schema()


def get_messages() -> List[Dict[str, Any]]:
    return session.setdefault("messages", [])


def get_history() -> List[Dict[str, Any]]:
    return session.setdefault("history", [])


def get_current_user() -> User | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def allowed_grades_for_user(user: User | None) -> List[int] | None:
    if not user or user.role != "user" or not user.student_grade:
        return None
    return [grade for grade in [4, 5, 6] if grade <= int(user.student_grade)]


def normalize_chat_provider(value: str | None) -> str:
    provider = (value or DEFAULT_CHAT_PROVIDER or "gemini").strip().lower()
    return provider if provider in {"gemini", "qwen"} else "gemini"


def get_chat_provider() -> str:
    provider = normalize_chat_provider(session.get("chat_provider"))
    session["chat_provider"] = provider
    return provider


def login_required(role: str | None = None):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            user = get_current_user()
            if not user:
                flash("Silakan login terlebih dahulu.", "error")
                return redirect(url_for("login_page"))
            if role and user.role != role:
                flash("Kamu tidak punya akses ke halaman tersebut.", "error")
                return redirect(url_for("chat_page"))
            return view_func(*args, **kwargs)

        return wrapped_view

    return decorator


def parse_heading_lines(raw_text: str) -> List[str]:
    headings: List[str] = []
    for line in (raw_text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        clean = clean.replace("\u2022", "").lstrip("-*").strip()
        if clean:
            headings.append(clean)
    return headings


def parse_heading_entries(form: Any) -> List[Dict[str, str]]:
    names = form.getlist("heading_name")
    descriptions = form.getlist("heading_description")
    section_modes = form.getlist("heading_mode")
    entries: List[Dict[str, str]] = []

    for index, raw_name in enumerate(names):
        name = (raw_name or "").strip()
        description = (descriptions[index] if index < len(descriptions) else "") or ""
        description = description.strip()
        section_mode = (section_modes[index] if index < len(section_modes) else "GENERAL") or "GENERAL"
        section_mode = section_mode.strip().upper()
        if section_mode not in {"GENERAL", "INSTRUKSI", "EVALUASI"}:
            section_mode = "GENERAL"
        if not name and not description:
            continue
        if not name:
            continue
        entries.append(
            {
                "name": name,
                "description": description,
                "section_mode": section_mode,
            }
        )

    return entries


def build_source_payload(src: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": src.get("chunk_id", "-"),
        "source_book": src.get("source_book", "-"),
        "heading": (src.get("headings") or ["Tanpa heading"])[0],
        "content": (src.get("content") or "").strip(),
    }


def summarize_sources_for_storage(sources: List[Dict[str, Any]]) -> str:
    payload = [build_source_payload(src) for src in sources[:5]]
    return json.dumps(payload, ensure_ascii=False)


def to_app_timezone(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=ZoneInfo("UTC")).astimezone(APP_TIMEZONE)
    return value.astimezone(APP_TIMEZONE)


def format_local_datetime(value: datetime | None, pattern: str, fallback: str = "") -> str:
    localized = to_app_timezone(value)
    if not localized:
        return fallback
    return localized.strftime(pattern)


def determine_bot_avatar_state(answer: str, sources: List[Dict[str, Any]], error: str | None = None) -> str:
    if error:
        return "sad"

    normalized_answer = (answer or "").strip().lower()
    if not normalized_answer:
        return "sad"

    sad_markers = [
        "maaf, jawaban belum ditemukan",
        "maaf, dari materi yang tersedia",
        "belum ditemukan",
        "belum ada pada materi",
        "belum ada di materi",
        "materi yang tersedia",
        "tidak ditemukan",
        "tidak ada informasi",
        "tidak ada pada materi",
        "tidak ada di materi",
        "tidak ditemukan pada materi",
        "tidak menjelaskan",
    ]
    if any(marker in normalized_answer for marker in sad_markers):
        return "sad"

    if not sources:
        return "sad"

    return "happy"


def save_chat_history_record(question: str, answer: str, sources: List[Dict[str, Any]]) -> None:
    current_user = get_current_user()
    if not current_user or current_user.role != "user":
        return

    db.session.add(
        ChatHistory(
            user_id=current_user.id,
            question=question,
            answer=answer,
            source_summary=summarize_sources_for_storage(sources),
        )
    )
    db.session.commit()


def build_chat_payload(question: str, provider: str | None = None) -> tuple[Dict[str, Any], Dict[str, Any], str | None]:
    now = datetime.now(APP_TIMEZONE)
    current_user = get_current_user()
    selected_provider = normalize_chat_provider(provider)
    user_message = {
        "role": "user",
        "text": question,
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%d-%m-%Y"),
    }

    try:
        answer, sources = generate_answer(
            question,
            allowed_grades=allowed_grades_for_user(current_user),
            provider=selected_provider,
        )
        error = None
    except Exception as exc:
        print("\n[WEBAPP ERROR]")
        print(f"Question: {question}")
        traceback.print_exc()
        answer = "Maaf, terjadi kendala saat memproses pertanyaanmu."
        sources = []
        error = str(exc)
    else:
        save_chat_history_record(question, answer, sources)

    bot_message = {
        "role": "assistant",
        "text": answer,
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%d-%m-%Y"),
        "sources": [build_source_payload(src) for src in sources],
        "provider": selected_provider,
        "avatar_state": determine_bot_avatar_state(answer, sources, error),
    }

    return user_message, bot_message, error


@app.route("/profile", methods=["GET", "POST"])
@login_required()
def profile_page():
    current_user = get_current_user()
    if not current_user:
        return redirect(url_for("login_page"))

    error = None
    success = None

    if request.method == "POST":
        current_password = request.form.get("current_password") or ""
        new_password = request.form.get("new_password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not check_password_hash(current_user.password_hash, current_password):
            error = "Password saat ini tidak sesuai."
        elif len(new_password) < 6:
            error = "Password baru minimal 6 karakter."
        elif new_password != confirm_password:
            error = "Konfirmasi password baru tidak cocok."
        elif check_password_hash(current_user.password_hash, new_password):
            error = "Password baru harus berbeda dari password saat ini."
        else:
            try:
                current_user.password_hash = generate_password_hash(new_password)
                db.session.commit()
                success = "Password berhasil diperbarui."
            except Exception as exc:
                db.session.rollback()
                error = f"Gagal memperbarui password: {exc}"

    return render_template(
        "profile.html",
        title="Profil Akun",
        brand_title="Bonibot/Profil",
        error=error,
        success=success,
        profile_user=current_user,
    )


def get_user_chat_history_items(user_id: int) -> List[Dict[str, Any]]:
    records = (
        ChatHistory.query.filter_by(user_id=user_id)
        .order_by(ChatHistory.created_at.desc())
        .all()
    )
    items: List[Dict[str, Any]] = []
    for record in records:
        items.append(
            {
                "id": record.id,
                "title": record.question[:50] + ("..." if len(record.question) > 50 else ""),
                "time": format_local_datetime(record.created_at, "%H:%M"),
                "date": format_local_datetime(record.created_at, "%d-%m-%Y"),
                "question": record.question,
                "answer": record.answer,
            }
        )
    return items


def get_history_user_summaries() -> List[Dict[str, Any]]:
    users = (
        User.query.filter_by(role="user")
        .order_by(User.created_at.desc())
        .all()
    )
    items: List[Dict[str, Any]] = []
    for user in users:
        latest_chat = (
            ChatHistory.query.filter_by(user_id=user.id)
            .order_by(ChatHistory.created_at.desc())
            .first()
        )
        items.append(
            {
                "user_id": user.id,
                "user_name": user.name,
                "username": user.username,
                "student_grade": user.student_grade,
                "created_at": format_local_datetime(user.created_at, "%d-%m-%Y %H:%M", "-"),
                "chat_count": len(user.chat_messages),
                "last_chat_at": format_local_datetime(latest_chat.created_at if latest_chat else None, "%d-%m-%Y %H:%M", "-"),
                "last_question": latest_chat.question[:70] + ("..." if len(latest_chat.question) > 70 else "") if latest_chat else None,
            }
        )
    return items


def run_pipeline_scripts() -> Dict[str, str]:
    print("\n[PIPELINE] Menjalankan main.py untuk ekstraksi, preprocessing, dan chunking...")
    main_run = subprocess.run(
        [sys.executable, "main.py"],
        cwd=PROJECT_ROOT,
        text=True,
        check=True,
    )
    print("[PIPELINE] main.py selesai.")

    print("[PIPELINE] Menjalankan build_index.py untuk embedding dan pembentukan index...")
    index_run = subprocess.run(
        [sys.executable, "build_index.py"],
        cwd=PROJECT_ROOT,
        text=True,
        check=True,
    )
    print("[PIPELINE] build_index.py selesai.")

    return {
        "title": "Proses indexing berhasil.",
        "details": (
            "Pipeline selesai dijalankan. "
            f"Kode keluar main.py={main_run.returncode}, build_index.py={index_run.returncode}."
        ),
    }


def save_book_to_database(metadata: Dict[str, Any], heading_entries: List[Dict[str, str]]) -> Book:
    current_user = get_current_user()
    book = Book(
        title=metadata["book_title"],
        subject=metadata["subject"],
        grade=metadata["grade"],
        pdf_filename=metadata["saved_pdf"],
        original_filename=metadata["original_filename"],
        notes=metadata["notes"],
        processing_status="uploaded",
        uploaded_by=current_user.id if current_user else None,
    )
    db.session.add(book)
    db.session.flush()

    for index, entry in enumerate(heading_entries, start=1):
        db.session.add(
            Heading(
                book_id=book.id,
                name=entry["name"],
                description=entry["description"],
                section_mode=entry["section_mode"],
                display_order=index,
            )
        )

    db.session.commit()
    return book


def find_metadata_path_for_book(book: Book) -> Path | None:
    for metadata_path in UPLOAD_METADATA_DIR.glob("*.json"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if data.get("saved_pdf") == book.pdf_filename:
            return metadata_path
    return None


def build_book_metadata(book: Book, heading_entries: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "uploaded_at": (book.created_at or datetime.now()).isoformat(timespec="seconds"),
        "book_title": book.title,
        "subject": book.subject,
        "grade": book.grade,
        "original_filename": book.original_filename or book.pdf_filename,
        "saved_pdf": book.pdf_filename,
        "notes": book.notes or "",
        "heading_count": len(heading_entries),
        "headings": [entry["name"] for entry in heading_entries],
        "heading_entries": heading_entries,
    }


def write_book_metadata_file(book: Book, heading_entries: List[Dict[str, str]]) -> Path:
    metadata_path = find_metadata_path_for_book(book)
    if metadata_path is None:
        metadata_path = UPLOAD_METADATA_DIR / f"{Path(book.pdf_filename).stem}.json"

    metadata_path.write_text(
        json.dumps(build_book_metadata(book, heading_entries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def replace_book_headings(book: Book, heading_entries: List[Dict[str, str]]) -> None:
    for heading in list(book.headings):
        db.session.delete(heading)

    db.session.flush()

    for index, entry in enumerate(heading_entries, start=1):
        db.session.add(
            Heading(
                book_id=book.id,
                name=entry["name"],
                description=entry["description"],
                section_mode=entry["section_mode"],
                display_order=index,
            )
        )


def sync_book_record(book: Book, form_data: Dict[str, str], heading_entries: List[Dict[str, str]]) -> None:
    book.title = form_data["book_title"] or book.title
    book.subject = form_data["subject"]
    book.grade = form_data["grade"]
    book.notes = form_data["notes"]
    book.processing_status = "uploaded"

    replace_book_headings(book, heading_entries)

    db.session.add(
        ProcessingLog(
            book_id=book.id,
            status="uploaded",
            message="Metadata buku diperbarui oleh guru. Silakan jalankan proses index ulang.",
        )
    )
    db.session.commit()
    write_book_metadata_file(book, heading_entries)


def delete_book_assets(book: Book) -> None:
    pdf_path = UPLOAD_DIR / book.pdf_filename
    if pdf_path.exists():
        pdf_path.unlink(missing_ok=True)

    metadata_path = find_metadata_path_for_book(book)
    if metadata_path and metadata_path.exists():
        metadata_path.unlink(missing_ok=True)


def build_legacy_metadata_from_filename(filename: str) -> Dict[str, Any] | None:
    stem = Path(filename).stem.strip()
    parts = stem.split("_")
    if len(parts) < 2:
        return None

    subject_code = parts[0].strip().upper()
    roman_grade = parts[-1].strip().upper()
    grade = ROMAN_GRADE_MAP.get(roman_grade)
    if not grade:
        return None

    subject_label = LEGACY_SUBJECT_LABELS.get(subject_code, subject_code)
    return {
        "title": f"{subject_label} Kelas {grade}",
        "subject": subject_label,
        "grade": grade,
        "pdf_filename": filename,
        "original_filename": filename,
        "notes": "Buku lama hasil sinkronisasi dari folder data.",
        "processing_status": "indexed",
    }


def sync_legacy_books_to_database() -> Dict[str, int]:
    added = 0
    skipped = 0

    if not LEGACY_BOOKS_DIR.exists():
        return {"added": 0, "skipped": 0}

    for pdf_path in sorted(LEGACY_BOOKS_DIR.glob("*.pdf")):
        existing_book = Book.query.filter_by(pdf_filename=pdf_path.name).first()
        if existing_book:
            skipped += 1
            continue

        metadata = build_legacy_metadata_from_filename(pdf_path.name)
        if not metadata:
            skipped += 1
            continue

        db.session.add(
            Book(
                title=metadata["title"],
                subject=metadata["subject"],
                grade=metadata["grade"],
                pdf_filename=metadata["pdf_filename"],
                original_filename=metadata["original_filename"],
                notes=metadata["notes"],
                processing_status=metadata["processing_status"],
                uploaded_by=None,
            )
        )
        added += 1

    db.session.commit()
    return {"added": added, "skipped": skipped}


def log_processing_result(status: str, message: str) -> None:
    books = (
        Book.query.filter(Book.processing_status.in_(["uploaded", "processing", "failed"]))
        .order_by(Book.created_at.desc())
        .all()
    )
    if not books:
        return

    for book in books:
        book.processing_status = status
        db.session.add(
            ProcessingLog(
                book_id=book.id,
                status=status,
                message=message[:5000] if message else None,
            )
        )

    db.session.commit()


def mark_books_processing() -> None:
    books = (
        Book.query.filter(Book.processing_status.in_(["uploaded", "failed"]))
        .order_by(Book.created_at.desc())
        .all()
    )
    if not books:
        return
    for book in books:
        book.processing_status = "processing"
    db.session.commit()


@app.route("/", methods=["GET", "POST"])
@login_required()
def chat_page():
    messages = get_messages()
    error = None
    current_provider = get_chat_provider()

    if request.method == "POST":
        question = (request.form.get("message") or "").strip()
        current_provider = normalize_chat_provider(request.form.get("provider"))
        session["chat_provider"] = current_provider
        if question:
            user_message, bot_message, error = build_chat_payload(question, provider=current_provider)
            messages.append(user_message)
            messages.append(bot_message)

            get_history().append(
                {
                    "title": question[:50] + ("..." if len(question) > 50 else ""),
                    "time": user_message["time"],
                    "date": user_message["date"],
                }
            )
            session.modified = True
            return redirect(url_for("chat_page"))

    if not messages:
        messages = [
            {
                "role": "assistant",
                "text": "Halo! Hari ini kita mau petualangan ilmu apa?",
                "time": "",
                "date": "",
                "sources": [],
                "avatar_state": "idle",
            }
        ]
        session["messages"] = messages

    return render_template("chat.html", messages=messages, error=error, current_provider=current_provider)


@app.route("/api/chat", methods=["POST"])
@login_required()
def chat_api():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    selected_provider = normalize_chat_provider(payload.get("provider"))
    session["chat_provider"] = selected_provider
    if not question:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400

    messages = get_messages()
    user_message, bot_message, error = build_chat_payload(question, provider=selected_provider)
    messages.append(user_message)
    messages.append(bot_message)
    get_history().append(
        {
            "title": question[:50] + ("..." if len(question) > 50 else ""),
            "time": user_message["time"],
            "date": user_message["date"],
        }
    )
    session.modified = True

    return jsonify(
        {
            "user_message": user_message,
            "bot_message": bot_message,
            "error": error,
            "provider": selected_provider,
        }
    )


@app.route("/history")
@login_required()
def history_page():
    current_user = get_current_user()
    if not current_user:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("login_page"))
    if current_user.role == "admin":
        return redirect(url_for("teacher_history_page"))
    return render_template("history.html", items=get_user_chat_history_items(current_user.id))


@app.route("/teacher/history")
@login_required("admin")
def teacher_history_page():
    return render_template(
        "teacher_history.html",
        title="History Chat",
        brand_title="Chatbot/History",
        users=get_history_user_summaries(),
    )


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/teacher/upload", methods=["GET", "POST"])
@login_required("admin")
def teacher_upload_page():
    success = None
    error = None
    process_result = None
    form_data = {
        "book_title": "",
        "subject": "",
        "grade": "",
        "notes": "",
        "heading_entries": [
            {"name": "", "description": "", "section_mode": "GENERAL"},
            {"name": "", "description": "", "section_mode": "GENERAL"},
            {"name": "", "description": "", "section_mode": "GENERAL"},
        ],
    }

    if request.method == "POST":
        action = (request.form.get("action") or "upload").strip().lower()

        if action == "process_index":
            try:
                mark_books_processing()
                process_result = run_pipeline_scripts()
                log_processing_result("indexed", process_result["details"])
            except subprocess.CalledProcessError as exc:
                error = (
                    "Proses indexing gagal dijalankan. "
                    + ((exc.stderr or exc.stdout or str(exc)).strip() or "Periksa log pipeline.")
                )
                log_processing_result("failed", error)
            except Exception as exc:
                error = f"Proses indexing gagal dijalankan: {exc}"
                log_processing_result("failed", error)

            return render_template(
                "teacher_upload.html",
                title="Upload Buku Guru",
                brand_title="Chatbot/Guru",
                success=success,
                error=error,
                process_result=process_result,
                form_data=form_data,
            )

        heading_entries = parse_heading_entries(request.form)
        form_data = {
            "book_title": (request.form.get("book_title") or "").strip(),
            "subject": (request.form.get("subject") or "").strip(),
            "grade": (request.form.get("grade") or "").strip(),
            "notes": (request.form.get("notes") or "").strip(),
            "heading_entries": heading_entries
            if heading_entries
            else [
                {"name": "", "description": "", "section_mode": "GENERAL"},
                {"name": "", "description": "", "section_mode": "GENERAL"},
                {"name": "", "description": "", "section_mode": "GENERAL"},
            ],
        }
        headings = [entry["name"] for entry in heading_entries]
        pdf_file = request.files.get("book_file")

        if not pdf_file or not pdf_file.filename:
            error = "File PDF buku belum dipilih."
        elif not pdf_file.filename.lower().endswith(".pdf"):
            error = "File yang diunggah harus berupa PDF."
        elif get_upload_file_size(pdf_file) > MAX_UPLOAD_PDF_BYTES:
            file_size = format_file_size(get_upload_file_size(pdf_file))
            error = (
                f"Ukuran PDF terlalu besar ({file_size}). "
                f"Maksimal upload adalah {MAX_UPLOAD_PDF_MB} MB per file."
            )
        elif not headings:
            error = "Daftar heading belum diisi."
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = secure_filename(pdf_file.filename) or f"buku_{timestamp}.pdf"
            pdf_filename = f"{timestamp}_{safe_name}"
            pdf_path = UPLOAD_DIR / pdf_filename
            pdf_file.save(pdf_path)

            metadata = {
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                "book_title": form_data["book_title"] or safe_name.rsplit(".", 1)[0],
                "subject": form_data["subject"],
                "grade": form_data["grade"],
                "original_filename": pdf_file.filename,
                "saved_pdf": pdf_filename,
                "notes": form_data["notes"],
                "heading_count": len(headings),
                "headings": headings,
                "heading_entries": heading_entries,
            }
            metadata_filename = f"{timestamp}_{Path(safe_name).stem}.json"
            metadata_path = UPLOAD_METADATA_DIR / metadata_filename
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            try:
                db_book = save_book_to_database(metadata, heading_entries)
            except Exception as exc:
                db.session.rollback()
                if pdf_path.exists():
                    pdf_path.unlink(missing_ok=True)
                if metadata_path.exists():
                    metadata_path.unlink(missing_ok=True)
                error = f"Gagal menyimpan data buku ke database: {exc}"
                return render_template(
                    "teacher_upload.html",
                    title="Upload Buku Guru",
                    brand_title="Chatbot/Guru",
                    success=None,
                    error=error,
                    process_result=None,
                    form_data=form_data,
                )

            success = {
                "book_title": metadata["book_title"],
                "filename": pdf_filename,
                "heading_count": len(headings),
                "metadata_file": metadata_filename,
                "book_id": db_book.id,
            }

            form_data = {
                "book_title": "",
                "subject": "",
                "grade": "",
                "notes": "",
                "heading_entries": [
                    {"name": "", "description": "", "section_mode": "GENERAL"},
                    {"name": "", "description": "", "section_mode": "GENERAL"},
                    {"name": "", "description": "", "section_mode": "GENERAL"},
                ],
            }

    return render_template(
        "teacher_upload.html",
        title="Upload Buku Guru",
        brand_title="Chatbot/Guru",
        success=success,
        error=error,
        process_result=process_result,
        form_data=form_data,
    )


@app.route("/teacher/books")
@login_required("admin")
def teacher_books_page():
    books = Book.query.order_by(Book.created_at.desc()).all()
    return render_template(
        "teacher_books.html",
        title="Daftar Buku Guru",
        brand_title="Chatbot/Buku Guru",
        books=books,
    )


@app.route("/teacher/books/sync-legacy", methods=["POST"])
@login_required("admin")
def teacher_sync_legacy_books_page():
    try:
        result = sync_legacy_books_to_database()
        flash(
            f"Sinkronisasi buku lama selesai. {result['added']} buku ditambahkan, {result['skipped']} dilewati.",
            "success",
        )
    except Exception as exc:
        db.session.rollback()
        flash(f"Gagal menyinkronkan buku lama: {exc}", "error")

    return redirect(url_for("teacher_books_page"))


@app.route("/teacher/books/<int:book_id>/edit", methods=["GET", "POST"])
@login_required("admin")
def teacher_edit_book_page(book_id: int):
    book = db.session.get(Book, book_id)
    if not book:
        flash("Buku tidak ditemukan.", "error")
        return redirect(url_for("teacher_books_page"))

    error = None
    form_data = {
        "book_title": book.title,
        "subject": book.subject,
        "grade": book.grade,
        "notes": book.notes or "",
        "heading_entries": [
            {
                "name": heading.name,
                "description": heading.description or "",
                "section_mode": heading.section_mode,
            }
            for heading in sorted(book.headings, key=lambda item: item.display_order)
        ]
        or [
            {"name": "", "description": "", "section_mode": "GENERAL"},
        ],
    }

    if request.method == "POST":
        heading_entries = parse_heading_entries(request.form)
        form_data = {
            "book_title": (request.form.get("book_title") or "").strip(),
            "subject": (request.form.get("subject") or "").strip(),
            "grade": (request.form.get("grade") or "").strip(),
            "notes": (request.form.get("notes") or "").strip(),
        }

        if not form_data["book_title"]:
            error = "Judul buku tidak boleh kosong."
        elif not form_data["subject"]:
            error = "Mata pelajaran belum diisi."
        elif not form_data["grade"]:
            error = "Kelas belum diisi."
        elif not heading_entries:
            error = "Minimal satu heading perlu diisi."
        else:
            try:
                sync_book_record(book, form_data, heading_entries)
                flash("Data buku berhasil diperbarui. Jalankan proses index ulang agar knowledge base ikut terbarui.", "success")
                return redirect(url_for("teacher_books_page"))
            except Exception as exc:
                db.session.rollback()
                error = f"Gagal memperbarui buku: {exc}"

    return render_template(
        "teacher_edit_book.html",
        title="Edit Buku Guru",
        brand_title="Chatbot/Edit Buku",
        book=book,
        error=error,
        form_data=form_data,
    )


@app.route("/teacher/books/<int:book_id>/delete", methods=["POST"])
@login_required("admin")
def teacher_delete_book_page(book_id: int):
    book = db.session.get(Book, book_id)
    if not book:
        flash("Buku tidak ditemukan.", "error")
        return redirect(url_for("teacher_books_page"))

    try:
        delete_book_assets(book)
        db.session.delete(book)
        db.session.commit()
        flash("Buku berhasil dihapus dari sistem guru.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Gagal menghapus buku: {exc}", "error")

    return redirect(url_for("teacher_books_page"))


@app.route("/teacher/users")
@login_required("admin")
def teacher_users_page():
    users = (
        User.query.filter_by(role="user")
        .order_by(User.created_at.desc())
        .all()
    )
    return render_template(
        "teacher_users.html",
        title="Daftar Siswa",
        brand_title="Chatbot/User",
        users=users,
    )


@app.route("/teacher/users/<int:user_id>/history")
@login_required("admin")
def teacher_user_history_page(user_id: int):
    user = User.query.filter_by(id=user_id, role="user").first()
    if not user:
        flash("User/siswa tidak ditemukan.", "error")
        return redirect(url_for("teacher_users_page"))

    items = get_user_chat_history_items(user.id)
    return render_template(
        "teacher_user_history.html",
        title="History Siswa",
        brand_title="Chatbot/History User",
        target_user=user,
        items=items,
    )


@app.route("/teacher/users/<int:user_id>/history/delete", methods=["POST"])
@login_required("admin")
def teacher_delete_user_history_page(user_id: int):
    user = User.query.filter_by(id=user_id, role="user").first()
    if not user:
        flash("User/siswa tidak ditemukan.", "error")
        return redirect(url_for("teacher_users_page"))

    try:
        ChatHistory.query.filter_by(user_id=user.id).delete()
        db.session.commit()
        flash(f"Seluruh history chat milik {user.name} berhasil dihapus.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Gagal menghapus history chat: {exc}", "error")

    return redirect(url_for("teacher_user_history_page", user_id=user.id))


@app.route("/teacher/users/<int:user_id>/delete", methods=["POST"])
@login_required("admin")
def teacher_delete_user_page(user_id: int):
    user = User.query.filter_by(id=user_id, role="user").first()
    if not user:
        flash("User/siswa tidak ditemukan.", "error")
        return redirect(url_for("teacher_users_page"))

    try:
        db.session.delete(user)
        db.session.commit()
        flash("Akun siswa beserta history chat-nya berhasil dihapus.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Gagal menghapus user: {exc}", "error")

    return redirect(url_for("teacher_users_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if get_current_user():
        return redirect(url_for("chat_page"))

    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            error = "Username atau password salah."
        elif not user.is_active:
            error = "Akun ini sedang tidak aktif."
        else:
            session.clear()
            session["user_id"] = user.id
            session["role"] = user.role
            session["display_name"] = user.name
            flash(f"Selamat datang, {user.name}.", "success")
            if user.role == "admin":
                return redirect(url_for("teacher_books_page"))
            return redirect(url_for("chat_page"))

    return render_template(
        "login.html",
        title="Login",
        brand_title="Chatbot/Login",
        error=error,
    )


@app.route("/register", methods=["GET", "POST"])
def register_page():
    if get_current_user():
        return redirect(url_for("chat_page"))

    error = None
    form_data = {
        "name": "",
        "username": "",
        "student_grade": "",
    }

    if request.method == "POST":
        form_data = {
            "name": (request.form.get("name") or "").strip(),
            "username": (request.form.get("username") or "").strip(),
            "student_grade": (request.form.get("student_grade") or "").strip(),
        }
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not form_data["name"]:
            error = "Nama lengkap belum diisi."
        elif not form_data["username"]:
            error = "Username belum diisi."
        elif len(form_data["username"]) < 4:
            error = "Username minimal 4 karakter."
        elif User.query.filter_by(username=form_data["username"]).first():
            error = "Username sudah digunakan. Silakan pilih username lain."
        elif form_data["student_grade"] not in {"4", "5", "6"}:
            error = "Silakan pilih kelas 4, 5, atau 6."
        elif len(password) < 6:
            error = "Password minimal 6 karakter."
        elif password != confirm_password:
            error = "Konfirmasi password tidak cocok."
        else:
            try:
                user = User(
                    name=form_data["name"],
                    username=form_data["username"],
                    password_hash=generate_password_hash(password),
                    role="user",
                    is_active=True,
                    student_grade=int(form_data["student_grade"]),
                )
                db.session.add(user)
                db.session.commit()
                flash("Akun berhasil dibuat. Silakan login menggunakan akun baru.", "success")
                return redirect(url_for("login_page"))
            except Exception as exc:
                db.session.rollback()
                error = f"Gagal membuat akun: {exc}"

    return render_template(
        "register.html",
        title="Register",
        brand_title="Chatbot/Register",
        error=error,
        form_data=form_data,
    )


@app.route("/logout")
def logout_page():
    session.clear()
    return redirect(url_for("login_page"))


@app.context_processor
def inject_user_context():
    return {
        "current_user": get_current_user(),
    }


@app.before_request
def redirect_root_for_admin():
    return None


if __name__ == "__main__":
    app.run(debug=True)
