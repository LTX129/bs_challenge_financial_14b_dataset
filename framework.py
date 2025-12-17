import os
from pathlib import Path

from rag.rag import RagManager
from rag.pdf_processor import PDFProcessor
from rag.sqlite_processor import SQLiteProcessor
from utils.util import get_model_bundle

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "dataset/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "langchain")
CHROMA_SERVER_TYPE = os.getenv("CHROMA_SERVER_TYPE", "local")
PDF_DIR = os.getenv("PDF_DIR", "dataset/pdf")
PDF_TXT_DIR = os.getenv("PDF_TXT_DIR", "dataset/pdf_txt_file")
SQLITE_PATH = os.getenv("SQLITE_PATH")  # resolved dynamically


def _load_models():
    llm, _chat, embed = get_model_bundle()
    return llm, embed


def _resolve_sqlite_path(override: str | None = None) -> Path:
    """
    Resolve a usable SQLite path.
    Priority:
      1) explicit override
      2) SQLITE_PATH env
      3) first *.db under dataset/database
    """
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    if SQLITE_PATH:
        candidates.append(Path(SQLITE_PATH))

    db_root = Path("dataset") / "database"
    if db_root.exists():
        for db_file in sorted(db_root.glob("*.db")):
            candidates.append(db_file)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No SQLite db found. Set SQLITE_PATH or place a .db file under dataset/database/."
    )


def test_rag(question: str | None = None):
    llm, embed = _load_models()
    rag = RagManager(
        chroma_server_type=CHROMA_SERVER_TYPE,
        persist_path=CHROMA_PERSIST_PATH,
        collection_name=CHROMA_COLLECTION,
        llm=llm,
        embed=embed,
    )

    query = question
    result = rag.get_result(query)
    print(result)


def import_pdfs(
    directory: str | None = None,
    text_directory: str | None = PDF_TXT_DIR,
):
    _llm, embed = _load_models()
    pdf_processor = PDFProcessor(
        directory=directory or PDF_DIR,
        chroma_server_type=CHROMA_SERVER_TYPE,
        persist_path=CHROMA_PERSIST_PATH,
        collection_name=CHROMA_COLLECTION,
        embed=embed,
        text_directory=text_directory,
    )
    pdf_processor.process_pdfs()


def import_sqlite(
    db_path: str | None = None,
    tables: list[str] | None = None,
    row_limit: int | None = None,
):
    _llm, embed = _load_models()

    resolved_path = _resolve_sqlite_path(db_path)
    effective_row_limit = row_limit
    if effective_row_limit is None:
        env_limit = os.getenv("SQLITE_ROW_LIMIT")
        if env_limit:
            try:
                effective_row_limit = int(env_limit)
            except ValueError:
                effective_row_limit = None

    fetch_batch = int(os.getenv("SQLITE_FETCH_BATCH", "2000"))
    rows_per_doc = int(os.getenv("SQLITE_ROWS_PER_DOC", "20"))
    insert_batch = int(os.getenv("SQLITE_INSERT_BATCH", "64"))

    processor = SQLiteProcessor(
        db_path=str(resolved_path),
        chroma_server_type=CHROMA_SERVER_TYPE,
        persist_path=CHROMA_PERSIST_PATH,
        collection_name=CHROMA_COLLECTION,
        embed=embed,
        fetch_batch=fetch_batch,
        rows_per_doc=rows_per_doc,
        insert_batch_size=insert_batch,
        row_limit=effective_row_limit,
    )
    processor.process(tables=tables)


if __name__ == "__main__":
    test_rag()
