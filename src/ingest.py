import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from db import (
    get_connection,
    collection_has_any,
    collection_has_source,
    delete_source,
    get_vectorstore,
)
from common import ask_provider_interactively, pick_default_provider_from_env, validate_environment

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ------------------------
# Validação ingestão
# ------------------------
def validate_if_already_ingested(path: Path, collection: str, db_url: str) -> str:
    with get_connection() as conn:
        has_any = collection_has_any(conn, collection)
        has_this_pdf = collection_has_source(conn, collection, str(path))

    if has_this_pdf:
        resp = input(f"[ingest] Já existem vetores deste PDF em '{collection}'. Deseja reprocessar? [s/N]: ").strip().lower()
        return "delete" if resp in ("s", "sim", "y", "yes") else "skip"

    if has_any:
        Console().print(f"[ingest] A collection '{collection}' já contém vetores de outras fontes.")
        resp = input("Deseja adicionar (append) os vetores deste PDF? [s/N]: ").strip().lower()
        return "append" if resp in ("s", "sim", "y", "yes") else "skip"

    return "new"


# ------------------------
# Processamento PDF
# ------------------------
def get_documents_from_pdf(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    splits = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    ).split_documents(docs)
    return [Document(page_content=d.page_content, metadata=d.metadata) for d in splits]


def def_index_documents(docs: list[Document]) -> list[str]:
    return [f"doc-{i}" for i in range(len(docs))]


# ------------------------
# Ingestão
# ------------------------
def ingest(pdf_path: str | None = None):
    db_url = os.getenv("DATABASE_URL")
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
    path = Path(pdf_path or os.getenv("PDF_PATH", "./document.pdf")).resolve()

    action = validate_if_already_ingested(path, collection, db_url)
    if action == "skip":
        Console().print(Panel.fit("[ingest] Operação cancelada."))
        return
    if action == "delete":
        with get_connection() as conn:
            delete_source(conn, collection, str(path))

    docs = get_documents_from_pdf(path)
    provider = ask_provider_interactively(default=pick_default_provider_from_env(), sufixo="[Ingest]")
    store = get_vectorstore(provider)
    store.add_documents(documents=docs, ids=def_index_documents(docs))
    Console().print(Panel.fit(f"[ingest] Stored {len(docs)} chunks in collection '{collection}'."))


def main(pdf_path: str | None = None):
    validate_environment(["DATABASE_URL", "PDF_PATH", "PG_VECTOR_COLLECTION_NAME"])
    ingest(pdf_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a PDF into Postgres pgvector.")
    parser.add_argument("--pdf", help="Path to the PDF (default: ./document.pdf)")
    args = parser.parse_args()
    main(args.pdf)
