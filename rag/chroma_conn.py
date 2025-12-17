import chromadb
from chromadb import Settings
from langchain_chroma import Chroma


class ChromaDB:
    """
    Small helper around Chroma to keep configuration in one place.
    Supports both local persistence and HTTP mode.
    """

    def __init__(
        self,
        chroma_server_type: str = "local",
        host: str = "localhost",
        port: int = 8000,
        persist_path: str = "chroma_db",
        collection_name: str = "langchain",
        embed=None,
    ):
        self.host = host
        self.port = port
        self.path = persist_path
        self.embed = embed
        self.store = None

        client_settings = Settings(anonymized_telemetry=False)

        if chroma_server_type == "http":
            client = chromadb.HttpClient(host=host, port=port, settings=client_settings)
            self.store = Chroma(
                collection_name=collection_name,
                embedding_function=embed,
                client=client,
            )
        elif chroma_server_type == "local":
            self.store = Chroma(
                collection_name=collection_name,
                embedding_function=embed,
                persist_directory=persist_path,
                client_settings=client_settings,
            )
        else:
            raise ValueError(f"Unsupported chroma_server_type: {chroma_server_type}")

        if self.store is None:
            raise ValueError("Chroma store initialization failed!")

    def add_with_langchain(self, docs):
        self.store.add_documents(documents=docs)

    def get_store(self):
        return self.store
