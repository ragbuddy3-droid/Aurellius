from datetime import datetime

from database import db


class TimestampMixin:
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class User(TimestampMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user", index=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    student_grade = db.Column(db.Integer, nullable=True)

    books = db.relationship("Book", back_populates="uploader", lazy=True)
    chat_messages = db.relationship(
        "ChatHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy=True,
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


class Book(TimestampMixin, db.Model):
    __tablename__ = "books"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    grade = db.Column(db.String(20), nullable=False)
    pdf_filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    processing_status = db.Column(db.String(30), nullable=False, default="uploaded")

    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    uploader = db.relationship("User", back_populates="books")
    headings = db.relationship(
        "Heading",
        back_populates="book",
        cascade="all, delete-orphan",
        lazy=True,
    )
    processing_logs = db.relationship(
        "ProcessingLog",
        back_populates="book",
        cascade="all, delete-orphan",
        lazy=True,
    )

    def __repr__(self) -> str:
        return f"<Book {self.title}>"


class Heading(TimestampMixin, db.Model):
    __tablename__ = "headings"

    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    section_mode = db.Column(db.String(20), nullable=False, default="GENERAL")
    display_order = db.Column(db.Integer, nullable=False, default=0)

    book = db.relationship("Book", back_populates="headings")

    def __repr__(self) -> str:
        return f"<Heading {self.name} ({self.section_mode})>"


class ChatHistory(TimestampMixin, db.Model):
    __tablename__ = "chat_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    source_summary = db.Column(db.Text, nullable=True)

    user = db.relationship("User", back_populates="chat_messages")

    def __repr__(self) -> str:
        return f"<ChatHistory user_id={self.user_id} id={self.id}>"


class ProcessingLog(TimestampMixin, db.Model):
    __tablename__ = "processing_logs"

    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey("books.id"), nullable=False, index=True)
    status = db.Column(db.String(30), nullable=False, default="pending")
    message = db.Column(db.Text, nullable=True)

    book = db.relationship("Book", back_populates="processing_logs")

    def __repr__(self) -> str:
        return f"<ProcessingLog book_id={self.book_id} status={self.status}>"
