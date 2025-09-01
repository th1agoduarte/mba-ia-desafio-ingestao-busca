import os
import psycopg
from typing import Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from rich.console import Console
from rich.panel import Panel
from common import normalize_for_psycopg,ask_provider_interactively, get_embeddings, pick_default_provider_from_env, validate_environment

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ------------------------
# Helpers de banco
# ------------------------
def _get_collection_id(conn: psycopg.Connection, collection_name: str) -> Optional[str]:
    row = conn.execute(
        "SELECT uuid FROM langchain_pg_collection WHERE name = %s LIMIT 1",
        (collection_name,),
    ).fetchone()
    return row[0] if row else None


def already_ingested_any(conn: psycopg.Connection, collection_name: str) -> bool:
    cid = _get_collection_id(conn, collection_name)
    if not cid:
        return False
    row = conn.execute(
        "SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s",
        (cid,),
    ).fetchone()
    return (row[0] or 0) > 0


def already_ingested_this_pdf(conn: psycopg.Connection, collection_name: str, source_path: str) -> bool:
    cid = _get_collection_id(conn, collection_name)
    if not cid:
        return False
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM langchain_pg_embedding e
        WHERE e.collection_id = %s
          AND e.cmetadata->>'source' = %s
        """,
        (cid, source_path),
    ).fetchone()
    return (row[0] or 0) > 0


def delete_by_source(conn: psycopg.Connection, collection_name: str, source_path: str):
    """
    Apaga todos os vetores de um PDF específico (metadata.source).
    """
    cid = _get_collection_id(conn, collection_name)
    if not cid:
        return
    conn.execute(
        """
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = %s
          AND cmetadata->>'source' = %s
        """,
        (cid, source_path),
    )
    conn.commit()
    print(f"[ingest] Vetores anteriores do PDF '{source_path}' removidos da collection '{collection_name}'.")

# ------------------------
# Lógica de validação
# ------------------------
def prompt_overwrite(message: str) -> bool:
    while True:
        ans = input(f"{message} [s/N]: ").strip().lower()
        if ans in ("s", "sim", "y", "yes"):
            return True
        if ans in ("", "n", "nao", "não", "no"):
            return False
        print("Responda com 's' ou 'n'.")

def validate_if_already_ingested(path: Path, collection: str, db_url: str) -> str:
    """
    Retorna:
    - 'skip'   se o usuário cancelar
    - 'delete' se for reprocessar o mesmo PDF
    - 'append' se for adicionar a uma collection já existente
    - 'new'    se não havia nada
    """
    db_url = normalize_for_psycopg(db_url)
    with psycopg.connect(db_url) as conn:
        has_any = already_ingested_any(conn, collection)
        has_this_pdf = already_ingested_this_pdf(conn, collection, str(path))

    if has_this_pdf:
        if prompt_overwrite(
            f"[ingest] Já existem vetores deste PDF em '{collection}'. Deseja reprocessar?"
        ):
            return "delete"
        return "skip"
    if has_any:
        Console().print(f"[ingest] A collection '{collection}' já contém vetores de outras fontes.")
        if prompt_overwrite("Deseja adicionar (append) os vetores deste PDF?"):
            return "append"
        return "skip"
    return "new"

# ------------------------
# Ingestão
# ------------------------
def ingest(pdf_path: str | None = None):
    db_url = os.getenv("DATABASE_URL")
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
    path = Path(pdf_path or os.getenv("PDF_PATH", "./document.pdf")).resolve()

    action = validate_if_already_ingested(path, collection, db_url)
    if action == "skip":
        Console().print(Panel.fit("[ingest] Operação cancelada. Nada foi alterado."))
        return
    if action == "delete":
        with psycopg.connect(normalize_for_psycopg(db_url)) as conn:
            delete_by_source(conn, collection, str(path))

    docs = get_documents_from_pdf(path)

    default_provider = pick_default_provider_from_env()
    provider = ask_provider_interactively(default=default_provider, sufixo="[Ingest]")
    store = PGVector(
        embeddings=get_embeddings(provider),
        collection_name=collection,
        connection=db_url,
        use_jsonb=True,
    )
    store.add_documents(documents=docs, ids=def_index_documents(docs))
    Console().print(Panel.fit(f"[ingest] Stored {len(docs)} chunks in collection '{collection}'."))

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
    if not splits:
        raise SystemExit("No document splits were created.")
    return [
        Document(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)},
        )
        for doc in splits
    ]

def def_index_documents(enriched: list[Document]) -> list[str]:
    return [f"doc-{i}" for i in range(len(enriched))]

# ------------------------
# CLI
# ------------------------
def main(pdf_path: str | None = None):
    required_vars = ["DATABASE_URL", "PDF_PATH", "PG_VECTOR_COLLECTION_NAME"]
    validate_environment(required_vars)
    ingest(pdf_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a PDF into Postgres pgvector via LangChain.")
    parser.add_argument("--pdf", help="Path to the PDF (default: PDF_PATH or ./document.pdf)")
    args = parser.parse_args()
    main(args.pdf)
