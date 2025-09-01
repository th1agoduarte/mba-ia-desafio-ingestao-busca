import os
from rich.console import Console
from rich.panel import Panel

from db import get_connection, get_collection_id, collection_has_any, get_vectorstore
from search import similarity_search_with_score
from ingest import ingest
from common import ask_provider_interactively, get_llm, pick_default_provider_from_env, load_prompt_text, validate_environment


def build_context(docs_with_scores):
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Chunk {i} | page {page} | score {score:.4f}]\n{doc.page_content}")
    return "\n\n".join(parts)


def check_if_collection_empty():
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
    with get_connection() as conn:
        cid = get_collection_id(conn, collection)
        return not cid or not collection_has_any(conn, collection)


def main():
    validate_environment()
    console = Console()

    if check_if_collection_empty():
        console.print("[yellow]Nenhum dado encontrado na collection.[/yellow]")
        resp = console.input("Deseja rodar a ingestão agora? [s/N]: ").strip().lower()
        if resp in ("s", "sim", "y", "yes"):
            ingest()
        else:
            console.print("[red]Sem dados no banco. Encerrando o chat.[/red]")
            return
    provider = ask_provider_interactively(default=pick_default_provider_from_env(), sufixo="[Chat]")

    console.print(Panel.fit("Selecione o prompt: 1=PT, 2=EN. ENTER=PT"))
    sel = console.input("> ").strip()
    prompt_path = "./prompts/qa_prompt_pt.txt" if sel != "2" else "./prompts/qa_prompt_en.txt"
    template = load_prompt_text(prompt_path)

    llm = get_llm(provider)
    console.print(Panel.fit("Faça sua pergunta (ou 'sair'/'exit')"))

    while True:
        try:
            question = console.input("[bold]PERGUNTA:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nSaindo...")
            break

        if not question or question.lower() in {"sair", "exit", "quit"}:
            console.print("Saindo...")
            break

        results = similarity_search_with_score(question, k=os.getenv("k_RELEVANT_DOCUMENTS", 10), provider=provider)
        if not results:
            console.print("[yellow]Não tenho informações necessárias para responder sua pergunta.[/yellow]")
            continue

        context = build_context(results)
        prompt = template.format(context=context, question=question)
        # console.print(Panel(prompt, title="PROMPT ENVIADO", expand=False))

        try:
            ai_msg = llm.invoke(prompt)
            answer = getattr(ai_msg, "content", str(ai_msg))
        except Exception as e:
            answer = f"Erro ao consultar a LLM: {e}"

        console.print(Panel(answer, title="RESPOSTA", expand=False))


if __name__ == "__main__":
    main()
