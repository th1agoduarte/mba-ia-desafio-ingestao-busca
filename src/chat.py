import os

from rich.console import Console
from rich.panel import Panel

from search import similarity_search_with_score
from common import (
    ask_provider_interactively,
    get_llm,
    pick_default_provider_from_env,
    load_prompt_text,
    validate_environment,
)

# >>> import extra do ingest para poder rodar ingestão caso necessário
from ingest import ingest, _get_collection_id, already_ingested_any
import psycopg


def build_context(docs_with_scores):
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        page = meta.get("page", "N/A")
        parts.append(f"[Chunk {i} | page {page} | score {score:.4f}]\n{doc.page_content}")
    return "\n\n".join(parts)


def check_if_collection_empty():
    """
    Verifica se a collection tem vetores. Se não tiver, retorna True (está vazia).
    """
    db_url = os.getenv("DATABASE_URL")
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
    if not db_url:
        raise ValueError("DATABASE_URL não configurada no .env")

    # normaliza URL para psycopg
    if db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    with psycopg.connect(db_url) as conn:
        cid = _get_collection_id(conn, collection)
        if not cid:
            return True
        return not already_ingested_any(conn, collection)


def main():
    validate_environment()
    console = Console()

    # >>> NOVO: checar se já existe ingestão
    if check_if_collection_empty():
        console.print("[yellow]Nenhum dado encontrado na collection.[/yellow]")
        resp = console.input("Deseja rodar a ingestão agora? [s/N]: ").strip().lower()
        if resp in ("s", "sim", "y", "yes"):
            ingest()  # roda ingestão com o PDF padrão ou do .env
        else:
            console.print("[red]Sem dados no banco. Encerrando o chat.[/red]")
            return

    default_provider = pick_default_provider_from_env()
    provider = ask_provider_interactively(default=default_provider, sufixo="[Chat]")

    console.print(Panel.fit("Selecione o prompt: 1=PT (./prompts/qa_prompt_pt.txt), 2=EN (./prompts/qa_prompt_en.txt). ENTER=PT"))
    sel = console.input("> ").strip()
    prompt_path = "./prompts/qa_prompt_pt.txt" if sel != "2" else "./prompts/qa_prompt_en.txt"

    try:
        template = load_prompt_text(prompt_path)
    except Exception as e:
        console.print(f"[red]Erro ao carregar prompt ({prompt_path}): {e}[/red]")
        return

    llm = get_llm(provider)
    console.print(Panel.fit("Faça sua pergunta (ou 'sair'/'exit') / Ask your question (or 'exit')"))

    while True:
        try:
            question = console.input("[bold]PERGUNTA:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nSaindo.../Exiting...")
            break

        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            console.print("Saindo.../Exiting...")
            break
        
        results = similarity_search_with_score(question, k=10, provider=provider)
        
        if not results:
            console.print("[yellow]Não tenho informações necessárias para responder sua pergunta.[/yellow]")
            continue

        context = build_context(results)
        prompt = template.format(context=context, question=question)
        console.print(Panel(prompt, title="PROMPT ENVIADO / PROMPT SENT", expand=False))
        try:
            ai_msg = llm.invoke(prompt)
            answer = getattr(ai_msg, "content", str(ai_msg))
        except Exception as e:
            answer = f"Erro ao consultar a LLM: {e}"

        console.print(Panel(answer, title="RESPOSTA / ANSWER", expand=False))


if __name__ == "__main__":
    main()
