import os
import shutil
from pathlib import Path

from framework import import_sqlite, import_pdfs

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "dataset/chroma_db")


def main():
    """
    This script will completely rebuild the ChromaDB index.
    It first deletes the existing index, then re-imports all data from
    both SQLite and PDF files.
    """
    db_path = Path(CHROMA_PERSIST_PATH)
    if db_path.exists():
        print(f"Deleting existing ChromaDB at: {db_path}")
        shutil.rmtree(db_path)
        print("Deletion complete.")

    print("\nRe-indexing SQLite data...")
    import_sqlite()
    print("SQLite import complete.")

    print("\nRe-indexing PDF data...")
    import_pdfs()
    print("PDF import complete.")

    print("\nIndex rebuild is complete.")


if __name__ == "__main__":
    main()
