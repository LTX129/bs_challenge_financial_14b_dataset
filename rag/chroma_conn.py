import chromadb  # 导入 ChromaDB 的核心库
from chromadb import Settings  # 导入 ChromaDB 的配置类
from langchain_chroma import Chroma  # 导入 LangChain 的 Chroma 向量存储模块

class ChromaDB:
    """
    自定义的 ChromaDB 类，用于管理 Chroma 向量数据库。
    支持本地模式和 HTTP 模式连接 ChromaDB。
    """

    def __init__(self,
                 chroma_server_type="local",  # 服务器类型：http 表示通过 HTTP 连接，local 表示本地文件方式
                 host="localhost", port=8000,  # 如果是 HTTP 模式，需要指定服务器地址和端口
                 persist_path="chroma_db",  # 数据库路径：如果是本地模式，需要指定数据库的存储路径
                 collection_name="langchain",  # 数据库中的集合名称
                 embed=None  # 数据库使用的向量化函数
                 ):
        """
        初始化 ChromaDB 对象。

        :param chroma_server_type: 连接类型（"local" 或 "http"）
        :param host: HTTP 服务器地址（仅在 chroma_server_type="http" 时有效）
        :param port: HTTP 服务器端口（仅在 chroma_server_type="http" 时有效）
        :param persist_path: 数据库存储路径（仅在 chroma_server_type="local" 时有效）
        :param collection_name: 数据库集合名称
        :param embed: 向量化函数
        """
        self.host = host  # HTTP 服务器地址
        self.port = port  # HTTP 服务器端口
        self.path = persist_path  # 数据库存储路径
        self.embed = embed  # 向量化函数
        self.store = None  # 初始化时未创建向量存储对象

        # 如果是 HTTP 协议方式连接数据库
        if chroma_server_type == "http":
            # 创建一个 HTTP 客户端
            client = chromadb.HttpClient(host=host, port=port)
            # 初始化 Chroma 向量存储对象
            self.store = Chroma(collection_name=collection_name,
                                embedding_function=embed,
                                client=client)

        # 如果是本地模式连接数据库
        elif chroma_server_type == "local":
            # 初始化 Chroma 向量存储对象
            self.store = Chroma(collection_name=collection_name,
                                embedding_function=embed,
                                persist_directory=persist_path)

        # 如果初始化失败，抛出异常
        if self.store is None:
            raise ValueError("Chroma store initialization failed!")

    def add_with_langchain(self, docs):
        """
        将文档添加到 ChromaDB 数据库中。

        :param docs: 要添加的文档列表
        """
        # 使用 LangChain 提供的接口将文档添加到数据库
        self.store.add_documents(documents=docs)

    def get_store(self):
        """
        返回 Chroma 向量数据库的对象实例。

        :return: Chroma 向量存储对象
        """
        return self.store