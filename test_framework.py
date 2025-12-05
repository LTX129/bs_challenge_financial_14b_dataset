# 测试将 PDF 文件内容导入到向量数据库的主流程

def test_import():
    """
    测试从 PDF 文件加载内容并将其导入到 ChromaDB 向量数据库的过程
    """
    # 导入 PDFProcessor 类，用于处理 PDF 文件并将其内容存储到向量数据库中
    from rag.pdf_processor import PDFProcessor

    # 导入工具函数 get_qwen_models，用于获取通义千问的模型实例
    from utils.util import get_qwen_models

    # 获取通义千问的 LLM、Chat 和 Embedding 模型
    # 这里只使用 Embedding 模型来生成文本的向量表示
    llm, chat, embed = get_qwen_models()

    # 如果需要使用其他嵌入模型（如 Hugging Face 的 Embeddings），可以取消注释以下行
    # embed = get_huggingface_embeddings()

    # 定义 PDF 文件所在的目录路径
    directory = "./datasets/pdf"  # 存放 PDF 文件的目录

    # 定义 ChromaDB 数据库的持久化路径
    persist_path = "chroma_db"  # ChromaDB 数据库文件的存储路径

    # 定义 ChromaDB 的服务器类型（"local" 表示本地模式）
    server_type = "local"  # 使用本地模式连接 ChromaDB

    # 创建 PDFProcessor 实例，用于处理 PDF 文件并将其内容存储到 ChromaDB 中
    pdf_processor = PDFProcessor(
        directory=directory,  # PDF 文件目录
        chroma_server_type=server_type,  # ChromaDB 服务器类型
        persist_path=persist_path,  # ChromaDB 数据库的持久化路径
        embed=embed  # 嵌入模型，用于生成文本向量
    )

    # 调用 process_pdfs 方法，开始处理 PDF 文件并将内容插入到 ChromaDB
    pdf_processor.process_pdfs()

# 如果此脚本作为主程序运行，则执行测试导入流程
if __name__ == "__main__":
    test_import()