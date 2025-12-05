# 导入必要的库
from dotenv import load_dotenv  # 用于加载环境变量
import os  # 用于操作文件路径

# 获取当前文件所在的目录
current_dir = os.path.dirname(__file__)

# 构建到 conf/.qwen 的相对路径
# 这里通过 os.path.join 拼接路径，确保跨平台兼容性
conf_file_path_qwen = os.path.join(current_dir, '..', 'conf', '.qwen')

# 加载 .qwen 文件中的环境变量
# dotenv_path 参数指定了环境变量文件的路径
load_dotenv(dotenv_path=conf_file_path_qwen)

def get_qwen_models():
    """
    加载通义千问系列的大模型，包括 LLM、Chat 和 Embedding 模型。
    """
    # 加载 LLM 大模型（语言生成模型）
    # 使用 Tongyi 类来实例化一个通义千问模型
    from langchain_community.llms.tongyi import Tongyi
    llm = Tongyi(
        model="qwen3-max",  # 指定模型类型为 qwen-max，适合复杂任务
        # 控制输出的随机性，值越低越保守
        top_p=0.7,         # 核采样参数，控制生成文本的多样性
        # 最大生成的 token 数量
    )

    # 加载 Chat 大模型（对话模型）
    # 使用 ChatTongyi 类来实例化一个通义千问对话模型
    from langchain_community.chat_models import ChatTongyi
    chat = ChatTongyi(
        model="qwen3-max",  # 指定模型类型为 qwen-max，适合高质量对话
        # 温度更低以获得更确定的回复
        top_p=0.2,         # 控制对话生成的多样性
        # 最大生成的 token 数量
    )

    # 加载 Embedding 大模型（嵌入模型）
    # 使用 DashScopeEmbeddings 类来实例化一个通义千问嵌入模型
    from langchain_community.embeddings import DashScopeEmbeddings
    embed = DashScopeEmbeddings(
        model="text-embedding-v3"  # 指定嵌入模型版本
    )

    # 返回加载的三个模型
    return llm, chat, embed