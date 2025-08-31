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

def build_context(docs_with_scores):
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        page = meta.get("page", "N/A")
        parts.append(f"[Chunk {i} | page {page} | score {score:.4f}]\n{doc.page_content}")
    return "\n\n".join(parts)

def main():
    validate_environment()
    console = Console()

    default_provider = pick_default_provider_from_env()
    provider = ask_provider_interactively(default=default_provider)

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

if __name__ == "__main__":
    main()
