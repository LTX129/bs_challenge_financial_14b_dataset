import logging
import os
import time
from pathlib import Path

from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chroma_conn import ChromaDB


class PDFProcessor:
    """
    Load PDF (or cached TXT) files, chunk them, and push into ChromaDB.
    """

    def __init__(
        self,
        directory: str,
        chroma_server_type: str,
        persist_path: str,
        embed,
        collection_name: str = "langchain",
        text_directory: str | None = None,
    ) -> None:
        self.directory = directory
        self.text_directory = text_directory
        self.file_group_num = 80
        self.batch_num = 64
        self.chunksize = 1500
        self.overlap = 100

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

    def load_pdf_files(self):
        pdf_files = []
        for file in os.listdir(self.directory):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(self.directory, file))

        logging.info("Found %s PDF files.", len(pdf_files))
        return pdf_files

    def load_pdf_content(self, pdf_path):
        """
        Prefer cached TXT (much faster) when available, fall back to parsing PDF.
        """
        base_name = Path(pdf_path).stem
        if self.text_directory:
            txt_path = Path(self.text_directory) / f"{base_name}.txt"
            if txt_path.exists():
                logging.info("Loading cached txt for %s", base_name)
                loader = TextLoader(str(txt_path), autodetect_encoding=True)
                return loader.load()

        logging.info("Parsing PDF directly for %s", base_name)
        pdf_loader = PyMuPDFLoader(file_path=pdf_path)
        return pdf_loader.load()

    def split_text(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunksize,
            chunk_overlap=self.overlap,
            length_function=len,
            add_start_index=True,
        )
        docs = splitter.split_documents(documents)
        logging.info("Split text into %s chunks.", len(docs))
        return docs

    def insert_docs_chromadb(self, docs, batch_size=64):
        logging.info("Inserting %s documents into ChromaDB.", len(docs))

        start_time = time.time()
        total_docs_inserted = 0
        total_batches = (len(docs) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Inserting batches", unit="batch") as pbar:
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                self.chroma_db.add_with_langchain(batch)
                total_docs_inserted += len(batch)

                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    tpm = (total_docs_inserted / elapsed_time) * 60
                    pbar.set_postfix({"TPM": f"{tpm:.2f}"})

                pbar.update(1)

    def process_pdfs_group(self, pdf_files_group):
        pdf_contents = []

        for pdf_path in pdf_files_group:
            documents = self.load_pdf_content(pdf_path)
            pdf_contents.extend(documents)

        docs = self.split_text(pdf_contents)
        self.insert_docs_chromadb(docs, self.batch_num)

    def process_pdfs(self):
        pdf_files = self.load_pdf_files()
        group_num = self.file_group_num

        for i in range(0, len(pdf_files), group_num):
            pdf_files_group = pdf_files[i : i + group_num]
            self.process_pdfs_group(pdf_files_group)

        print("PDFs processed successfully!")
