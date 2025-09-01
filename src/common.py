import os
from dotenv import load_dotenv
from typing import Literal

Provider = Literal["openai", "google"]

def load_environment():
    load_dotenv()

def validate_environment(required_vars: list[str] = None):
    load_environment()
    if required_vars is None:
        return
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

def normalize_for_psycopg(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql://", 1)
    return url

def ask_provider_interactively(default: Provider | None = None, sufixo: str = "") -> Provider:
    choices = {"1": "openai", "2": "google"}
    prompt = f"{sufixo} Escolha o provedor / Choose provider [1=OpenAI, 2=Google]"
    if default:
        prompt += f" (ENTER={default})"
    prompt += ": "
    while True:
        try:
            sel = input(prompt).strip()
        except EOFError:
            sel = ""
        if not sel and default:
            return default
        if sel in choices:
            return choices[sel]
        if sel.lower() in {"openai", "google"}:
            return sel  # type: ignore[return-value]
        print(f"{sufixo}Opção inválida. Digite 1, 2, 'openai' ou 'google'.")

def get_embeddings(provider: Provider):
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"))
    
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

def get_llm(provider: Provider):
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    from langchain_openai import ChatOpenAI
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
    return ChatOpenAI(model=model, temperature=0)

def pick_default_provider_from_env() -> Provider | None:
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_google = bool(os.getenv("GOOGLE_API_KEY", "").strip())
    if has_openai and not has_google:
        return "openai"
    if has_google and not has_openai:
        return "google"
    return None

def load_prompt_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
