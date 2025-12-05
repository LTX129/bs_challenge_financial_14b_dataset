import logging  # 日志记录模块
from langchain_core.prompts import ChatPromptTemplate  # 用于构建对话提示模板
from langchain_core.runnables import RunnablePassthrough  # 用于传递输入数据的工具
from langchain_core.runnables.base import RunnableLambda  # 用于包装自定义函数为可运行对象
from langchain_core.output_parsers import StrOutputParser  # 用于解析输出为字符串
from .chroma_conn import ChromaDB  # 自定义的 ChromaDB 类

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

class RagManager:
    """
    RAG（检索增强生成）管理器类，用于管理和执行基于向量数据库的知识检索与生成任务。
    """

    def __init__(self,
                 chroma_server_type="http",  # ChromaDB 服务器类型（"http" 或 "local"）
                 host="localhost", port=8000,  # ChromaDB HTTP 服务器地址和端口
                 persist_path="chroma_db",  # ChromaDB 数据库持久化路径
                 llm=None, embed=None):  # LLM 模型和嵌入模型

        self.llm = llm  # 大语言模型（LLM）
        self.embed = embed  # 嵌入模型

        # 初始化 ChromaDB 并获取存储对象
        chrom_db = ChromaDB(chroma_server_type=chroma_server_type,
                            host=host, port=port,
                            persist_path=persist_path,
                            embed=embed)
        self.store = chrom_db.get_store()  # 获取 ChromaDB 的存储实例

    def get_chain(self, retriever):
        """
        构建并返回 RAG 查询链。

        :param retriever: 向量数据库的检索器
        :return: RAG 查询链
        """
        # 定义 RAG 系统的经典 Prompt 模板
        prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an assistant for question-answering tasks. Use the following pieces 
          of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
          Use three sentences maximum and keep the answer concise.
          Question: {question} 
          Context: {context} 
          Answer:""")
        ])

        # 将 `format_docs` 方法包装为 Runnable 对象
        format_docs_runnable = RunnableLambda(self.format_docs)

        # 构建 RAG 查询链
        rag_chain = (
                {"context": retriever | format_docs_runnable,  # 使用检索器获取上下文
                 "question": RunnablePassthrough()}  # 直接传递问题
                | prompt  # 应用 Prompt 模板
                | self.llm  # 使用 LLM 模型生成答案
                | StrOutputParser()  # 解析输出为字符串
        )

        return rag_chain

    def format_docs(self, docs):
        """
        格式化检索到的文档内容。

        :param docs: 检索到的文档列表
        :return: 格式化后的文档内容
        """
        # 记录检索到的文档数量
        logging.info(f"检索到资料文件个数：{len(docs)}")

        # 提取文档来源信息
        retrieved_files = "\n".join([doc.metadata["source"] for doc in docs])
        logging.info(f"资料文件分别是:\n{retrieved_files}")

        # 提取文档内容
        retrieved_content = "\n\n".join(doc.page_content for doc in docs)
        logging.info(f"检索到的资料为:\n{retrieved_content}")

        return retrieved_content  # 返回格式化后的文档内容

    def get_retriever(self, k=4, mutuality=0.3):
        """
        获取向量数据库的检索器。

        :param k: 检索返回的文档数量
        :param mutuality: 相似度阈值（分数阈值）
        :return: 检索器对象
        """
        retriever = self.store.as_retriever(
            search_type="similarity_score_threshold",  # 使用相似度分数阈值检索
            search_kwargs={"k": k, "score_threshold": mutuality}  # 检索参数
        )
        return retriever

    def get_result(self, question, k=4, mutuality=0.3):
        """
        执行 RAG 查询并返回结果。

        :param question: 用户提出的问题
        :param k: 检索返回的文档数量
        :param mutuality: 相似度阈值（分数阈值）
        :return: 查询结果（生成的答案）
        """
        # 获取检索器
        retriever = self.get_retriever(k, mutuality)

        # 获取 RAG 查询链
        rag_chain = self.get_chain(retriever)

        # 执行查询并返回结果
        return rag_chain.invoke(input=question)