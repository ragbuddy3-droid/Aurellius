import os

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()

CHUNK_JSONL_PATH = os.getenv("CHUNK_JSONL_PATH", "out/chunks_all.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "rag.index")
STORE_PATH = os.getenv("STORE_PATH", "rag_store.json")

LEGACY_INDEX_PATH = "ipas.index"
LEGACY_STORE_PATH = "ipas_store.json"

TOP_K = int(os.getenv("TOP_K", "3"))
PRE_K = int(os.getenv("PRE_K", "60"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.80"))
RETRIEVAL_DEBUG = os.getenv("RETRIEVAL_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"} #Buat munculin hasil retrieval, kalo mau dimatiin ubah parameter 1 jadi 0

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
DEFAULT_CHAT_PROVIDER = os.getenv("DEFAULT_CHAT_PROVIDER", "gemini").strip().lower()
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "96"))
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.0"))

APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "skripsi-chatbot-secret-key")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///app.db",
)
SQLALCHEMY_TRACK_MODIFICATIONS = False

MAX_UPLOAD_PDF_MB = int(os.getenv("MAX_UPLOAD_PDF_MB", "50"))
MAX_UPLOAD_PDF_BYTES = MAX_UPLOAD_PDF_MB * 1024 * 1024
