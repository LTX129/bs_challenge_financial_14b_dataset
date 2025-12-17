import logging
import sqlite3
from pathlib import Path

from tqdm import tqdm
from langchain_core.documents import Document

from rag.chroma_conn import ChromaDB


class SQLiteProcessor:
    """
    Stream SQLite rows, chunk them, and store them in ChromaDB.
    """

    def __init__(
        self,
        db_path: str,
        chroma_server_type: str,
        persist_path: str,
        embed,
        collection_name: str = "langchain",
        fetch_batch: int = 2000,
        rows_per_doc: int = 50,
        insert_batch_size: int = 64,
        row_limit: int | None = None,
    ) -> None:
        self.db_path = db_path
        self.fetch_batch = fetch_batch
        self.rows_per_doc = rows_per_doc
        self.insert_batch_size = insert_batch_size
        self.row_limit = row_limit

        self.chroma_db = ChromaDB(
            chroma_server_type=chroma_server_type,
            persist_path=persist_path,
            collection_name=collection_name,
            embed=embed,
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _connect(self):
        db_path = Path(self.db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite db not found: {db_path}")
        return sqlite3.connect(db_path)

    def _list_tables(self, cursor):
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        logging.info("Discovered %s tables.", len(tables))
        return tables

    def _get_columns(self, cursor, table_name):
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        return [row[1] for row in cursor.fetchall()]

    def _schema_to_doc(self, table_name, columns):
        col_str = ", ".join(columns)
        doc_text = f"数据库中的表: {table_name}, 列: {col_str}"
        return Document(
            page_content=doc_text,
            metadata={
                "source": f"sqlite://{table_name}",
                "type": "schema",
            },
        )

    def _row_group_to_doc(self, rows, columns, table_name):
        docs = []
        for i in range(0, len(rows), self.rows_per_doc):
            chunk = rows[i : i + self.rows_per_doc]
            if not chunk:
                continue

            text_rows = []
            row_ids = []
            for row in chunk:
                row_id = row[0]
                row_ids.append(row_id)
                values = [
                    f"{col}: {value}"
                    for col, value in zip(columns, row[1:])
                    if value not in (None, "")
                ]
                text_rows.append("; ".join(values))

            doc_text = f"表: {table_name}\n" + "\n---\n".join(text_rows)
            docs.append(
                Document(
                    page_content=doc_text,
                    metadata={
                        "source": f"sqlite://{table_name}",
                        "row_start": min(row_ids),
                        "row_end": max(row_ids),
                    },
                )
            )
        return docs

    def _insert_docs(self, docs):
        for i in range(0, len(docs), self.insert_batch_size):
            batch = docs[i : i + self.insert_batch_size]
            self.chroma_db.add_with_langchain(batch)

    def process(self, tables: list[str] | None = None):
        with self._connect() as conn:
            cursor = conn.cursor()
            target_tables = tables or self._list_tables(cursor)
            if not target_tables:
                raise ValueError("No tables found in SQLite database.")

            for table_name in target_tables:
                logging.info("Processing table: %s", table_name)
                columns = self._get_columns(cursor, table_name)

                # Index schema
                schema_doc = self._schema_to_doc(table_name, columns)
                self._insert_docs([schema_doc])

                # Index rows
                cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                total_rows = cursor.fetchone()[0]
                if self.row_limit:
                    total_rows = min(total_rows, self.row_limit)

                cursor.execute(f'SELECT rowid, * FROM "{table_name}" ORDER BY rowid')

                processed = 0
                with tqdm(total=total_rows, desc=table_name, unit="rows") as pbar:
                    while True:
                        rows = cursor.fetchmany(self.fetch_batch)
                        if not rows:
                            break

                        processed += len(rows)
                        pbar.update(len(rows))

                        docs = self._row_group_to_doc(rows, columns, table_name)
                        self._insert_docs(docs)

                        if self.row_limit and processed >= self.row_limit:
                            break
