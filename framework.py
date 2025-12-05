def test_rag():
    """
    测试 RAG（检索增强生成）系统的主流程。
    """
    # 导入 RagManager 类，用于管理和执行 RAG 查询任务
    from rag.rag import RagManager

    # 导入工具函数 get_qwen_models，用于获取通义千问的模型实例
    from utils.util import get_qwen_models

    # 获取通义千问的 LLM、Chat 和 Embedding 模型
    llm, chat, embed = get_qwen_models()

    # 创建 RagManager 实例，用于管理 RAG 系统
    # 参数说明：
    # - host: ChromaDB HTTP 服务器地址
    # - port: ChromaDB HTTP 服务器端口
    # - llm: 大语言模型（LLM）
    # - embed: 嵌入模型（Embedding Model）
    rag = RagManager(host="localhost", port=8000, llm=llm, embed=embed)

    # 执行 RAG 查询，传入用户提出的问题
    question = "景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么?"
    result = rag.get_result(question)

    # 输出查询结果
    print(result)


if __name__ == "__main__":
    # 执行 RAG 测试函数
    test_rag()

    # 批量导入 PDF 测试函数（可选）
    # 如果需要测试 PDF 文件批量导入功能，请取消以下行的注释
    # test_import()