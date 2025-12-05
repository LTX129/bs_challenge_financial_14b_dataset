import os  # 文件操作相关模块
import logging  # 日志记录模块
import time  # 时间操作模块
from tqdm import tqdm  # 进度条显示模块
from langchain_community.document_loaders import PyMuPDFLoader  # PDF文件加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切分工具
from rag.chroma_conn import ChromaDB  # 自定义的ChromaDB类

class PDFProcessor:
    """
    PDFProcessor 类用于处理PDF文件并将其内容存储到ChromaDB中。
    """

    def __init__(self,
                 directory,  # PDF文件所在目录
                 chroma_server_type,  # ChromaDB服务器类型（"local" 或 "http"）
                 persist_path,  # ChromaDB持久化路径
                 embed):  # 向量化函数

        self.directory = directory  # PDF文件存放目录
        self.file_group_num = 80  # 每组处理的文件数
        self.batch_num = 6  # 每次插入的批次数量

        self.chunksize = 500  # 切分文本的大小
        self.overlap = 100  # 切分文本的重叠大小

        # 初始化ChromaDB对象
        self.chroma_db = ChromaDB(chroma_server_type=chroma_server_type,
                                  persist_path=persist_path,
                                  embed=embed)

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为INFO
            format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
            datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
        )

    def load_pdf_files(self):
        """
        加载指定目录下的所有PDF文件。
        """
        pdf_files = []
        for file in os.listdir(self.directory):  # 遍历目录中的所有文件
            if file.lower().endswith('.pdf'):  # 筛选PDF文件
                pdf_files.append(os.path.join(self.directory, file))  # 构建完整路径

        logging.info(f"Found {len(pdf_files)} PDF files.")  # 记录日志
        return pdf_files

    def load_pdf_content(self, pdf_path):
        """
        使用PyMuPDFLoader读取PDF文件的内容。
        """
        pdf_loader = PyMuPDFLoader(file_path=pdf_path)  # 创建PDF加载器
        docs = pdf_loader.load()  # 加载PDF内容
        logging.info(f"Loading content from {pdf_path}.")  # 记录日志
        return docs

    def split_text(self, documents):
        """
        使用RecursiveCharacterTextSplitter将文档切分为小段。
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunksize,  # 每段文本的最大长度
            chunk_overlap=self.overlap,  # 段与段之间的重叠长度
            length_function=len,  # 使用字符串长度作为分割依据
            add_start_index=True,  # 添加起始索引信息
        )
        docs = text_splitter.split_documents(documents)  # 切分文档
        logging.info("Split text into smaller chunks with RecursiveCharacterTextSplitter.")  # 记录日志
        return docs

    def insert_docs_chromadb(self, docs, batch_size=6):
        """
        将文档分批插入到ChromaDB中。
        """
        logging.info(f"Inserting {len(docs)} documents into ChromaDB.")  # 记录日志

        start_time = time.time()  # 记录开始时间
        total_docs_inserted = 0  # 已插入的文档总数

        # 计算总批次
        total_batches = (len(docs) + batch_size - 1) // batch_size

        # 使用tqdm显示进度条
        with tqdm(total=total_batches, desc="Inserting batches", unit="batch") as pbar:
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]  # 获取当前批次的文档
                self.chroma_db.add_with_langchain(batch)  # 插入到ChromaDB
                total_docs_inserted += len(batch)  # 更新已插入的文档数量

                # 计算TPM（每分钟插入的文档数）
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:  # 防止除以零
                    tpm = (total_docs_inserted / elapsed_time) * 60
                    pbar.set_postfix({"TPM": f"{tpm:.2f}"})  # 更新进度条的后缀信息

                pbar.update(1)  # 更新进度条

    def process_pdfs_group(self, pdf_files_group):
        """
        处理一组PDF文件，包括加载内容、切分文本和插入数据库。
        """
        pdf_contents = []  # 存储所有PDF文件的内容

        for pdf_path in pdf_files_group:
            documents = self.load_pdf_content(pdf_path)  # 加载PDF内容
            pdf_contents.extend(documents)  # 将内容添加到列表中

        docs = self.split_text(pdf_contents)  # 切分文本
        self.insert_docs_chromadb(docs, self.batch_num)  # 插入到ChromaDB

    def process_pdfs(self):
        """
        批量处理目录下的所有PDF文件。
        """
        pdf_files = self.load_pdf_files()  # 加载所有PDF文件

        group_num = self.file_group_num  # 每组处理的文件数

        # 按组处理PDF文件
        for i in range(0, len(pdf_files), group_num):
            pdf_files_group = pdf_files[i:i + group_num]
            self.process_pdfs_group(pdf_files_group)

        print("PDFs processed successfully!")  # 提示处理完成