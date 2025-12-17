import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


def _load_envs() -> None:
    """
    Load both local settings (.env) and legacy qwen settings (.qwen) from conf/.
    """
    current_dir = Path(__file__).resolve().parent
    env_dir = current_dir.parent / "conf"

    for filename in [".env", ".qwen"]:
        env_path = env_dir / filename
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_envs()


def get_qwen_models():
    """
    Remote Qwen stack (kept for compatibility). Use ChatTongyi for both llm/chat
    so it works with chat prompts.
    """
    from langchain_community.chat_models import ChatTongyi
    from langchain_community.embeddings import DashScopeEmbeddings
    api_key = os.getenv("API_KEY", "sk-2b251b52112a458cbdef09cfed67fe43")

    chat = ChatTongyi(model="qwen3-max", top_p=0.2, api_key = api_key)
    embed = DashScopeEmbeddings(model="text-embedding-v3")
    # llm and chat are the same chat model to keep prompt compatibility.
    return chat, chat, embed


def get_local_embeddings():
    """
    Local embedding model, defaults to BGE small zh. Place the model in LOCAL_EMBED_MODEL_PATH to stay offline.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = (
        os.getenv("LOCAL_EMBED_MODEL_PATH")
        or os.getenv("LOCAL_EMBED_MODEL_NAME")
        or "BAAI/bge-small-zh-v1.5"
    )
    cache_folder = os.getenv("LOCAL_EMBED_CACHE_DIR")
    device = os.getenv("LOCAL_EMBED_DEVICE", "cpu")

    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=cache_folder,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_local_chat_model():
    """
    Local chat/LLM driver. Defaults to Ollama with gpt-oss:latest so we can stay fully local.
    """
    backend = os.getenv("LOCAL_LLM_BACKEND", "openai").lower()
    temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.3"))

    if backend == "ollama":

        model = os.getenv("LOCAL_LLM_MODEL", "gemma3:12b")
        base_url = os.getenv("LOCAL_LLM_BASE_URL")
        kwargs = {"model": model, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)


    if backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise ImportError(
                "Set LOCAL_LLM_BACKEND=ollama or install langchain-openai for openai-compatible endpoints."
            ) from exc

        model = os.getenv("LOCAL_LLM_MODEL", "qwen3-max")
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("API_KEY", "sk-2b251b52112a458cbdef09cfed67fe43")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LOCAL_LLM_BACKEND: {backend}")


def get_model_bundle(prefer_backend: str | None = None):
    """
    Choose the model stack. Defaults to local embeddings (even for remote LLM)
    to avoid Chroma dimension mismatch unless explicitly overridden.
    """
    backend = (prefer_backend or os.getenv("RAG_MODEL_BACKEND", "local")).lower()
    embed_backend = os.getenv("RAG_EMBED_BACKEND", "local").lower()

    if backend == "qwen":
        llm, chat, remote_embed = get_qwen_models()
        # Default to local embeddings so we match the existing Chroma collection.
        embed = remote_embed if embed_backend in ("qwen", "remote") else get_local_embeddings()
        return llm, chat, embed

    if backend == "local":
        chat = get_local_chat_model()
        embed = get_local_embeddings()
        # llm and chat share the same object for simplicity
        return chat, chat, embed

    raise ValueError(f"Unknown backend: {backend}")
