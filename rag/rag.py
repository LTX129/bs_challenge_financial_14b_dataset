import logging

from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda

from .chroma_conn import ChromaDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class RagManager:
    """
    Build the full RAG chain backed by ChromaDB and expose helper APIs for answers
    (optionally with source snippets).
    """

    def __init__(
        self,
        chroma_server_type: str = "local",
        host: str = "localhost",
        port: int = 8000,
        persist_path: str = "chroma_db",
        collection_name: str = "langchain",
        llm=None,
        embed=None,
    ):
        self.llm = llm
        self.embed = embed

        chrom_db = ChromaDB(
            chroma_server_type=chroma_server_type,
            host=host,
            port=port,
            persist_path=persist_path,
            collection_name=collection_name,
            embed=embed,
        )
        self.store = chrom_db.get_store()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    """You are an assistant for question-answering tasks. Use the following pieces
of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:""",
                )
            ]
        )

    def get_chain(self, retriever):
        format_docs_runnable = RunnableLambda(self.format_docs)

        rag_chain = (
            {
                "context": retriever | format_docs_runnable,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def format_docs(self, docs):
        logging.info("Retrieved %s documents.", len(docs))

        retrieved_files = "\n".join([doc.metadata.get("source", "unknown") for doc in docs])
        logging.info("Sources:\n%s", retrieved_files)

        retrieved_content = "\n\n".join(doc.page_content for doc in docs)
        logging.debug("Retrieved content preview:\n%s", retrieved_content[:2000])

        return retrieved_content

    def get_retriever(self, k=4, mutuality=0.3, search_kwargs: dict | None = None):
        """
        Build a retriever; when mutuality<=0 fall back to plain similarity (no threshold).
        """
        _search_kwargs = search_kwargs or {}
        _search_kwargs["k"] = k

        if mutuality and mutuality > 0:
            _search_kwargs["score_threshold"] = mutuality
            return self.store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=_search_kwargs,
            )
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs=_search_kwargs,
        )

    def _fetch_docs(self, retriever, question):
        """
        Try multiple retriever interfaces for cross-version compatibility.
        """
        docs = None
        last_error: Exception | None = None

        try:
            docs = retriever.get_relevant_documents(question)
        except Exception as exc:
            last_error = exc

        if docs is None:
            try:
                docs = retriever.invoke(question)  # LCEL-compatible
            except Exception as exc:
                last_error = exc

        if docs is None:
            try:
                docs = retriever._get_relevant_documents(
                    question, run_manager=None
                )
            except Exception as exc:
                last_error = exc

        if docs is None:
            raise RuntimeError(
                "Retriever failed. If you see an embedding dimension mismatch, ensure the query "
                "embedding model matches the collection that was indexed."
            ) from last_error

        return docs

    def get_result(self, question, k=4, mutuality=0.3):
        result = self.get_result_with_sources(question, k, mutuality)
        return result["answer"]

    def get_result_with_sources(
        self,
        question: str,
        k: int = 4,
        mutuality: float = 0.3,
        source_char_limit: int = 380,
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline and also return lightweight source snippets for UI display.
        This implementation uses a two-step retrieval process for SQLite sources.
        """
        # Step 1: Retrieve relevant table schemas
        schema_retriever = self.get_retriever(
            k=5, search_kwargs={"filter": {"type": "schema"}}
        )
        schema_docs = self._fetch_docs(schema_retriever, question)
        
        table_sources = [
            doc.metadata["source"] for doc in schema_docs if "sqlite" in doc.metadata.get("source", "")
        ]

        # Step 2: Retrieve data from the identified tables or fallback to general search
        if table_sources:
            logging.info(f"Found relevant tables: {table_sources}")
            # Deduplicate sources
            table_sources = list(set(table_sources))
            
            data_retriever = self.get_retriever(
                k=k,
                mutuality=mutuality,
                search_kwargs={"filter": {"source": {"$in": table_sources}}},
            )
            docs = self._fetch_docs(data_retriever, question)
        else:
            logging.info("No specific table schema found, falling back to general search.")
            retriever = self.get_retriever(k, mutuality)
            docs = self._fetch_docs(retriever, question)

        # If threshold search returned nothing, retry with plain similarity to avoid over-filtering.
        if not docs:
            logging.warning(
                "No docs found with threshold %s; retrying with plain similarity (k=%s).",
                mutuality,
                k,
            )
            similarity_retriever = self.get_retriever(k=k, mutuality=0)
            docs = self._fetch_docs(similarity_retriever, question)

        formatted_context = self.format_docs(docs)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": formatted_context})

        sources: List[Dict[str, Any]] = []
        for doc in docs:
            preview = doc.page_content[:source_char_limit]
            sources.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "preview": preview,
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "hits": len(docs),
        }
