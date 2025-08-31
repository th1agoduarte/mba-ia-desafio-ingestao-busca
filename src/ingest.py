import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector

from common import ask_provider_interactively, get_embeddings, pick_default_provider_from_env, validate_environment

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def main(pdf_path: str | None = None):
    required_vars = ["DATABASE_URL", "PDF_PATH", "PG_VECTOR_COLLECTION_NAME"]
    validate_environment(required_vars)
    ingest(pdf_path)  

def validate_pdf_path(pdf_path: str) -> Path:
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return path

def ingest(pdf_path: str | None = None):
    default_provider = pick_default_provider_from_env()
    provider = ask_provider_interactively(default=default_provider)
    path = validate_pdf_path(pdf_path or os.getenv("PDF_PATH", "./document.pdf"))
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
    docs = get_documents_from_pdf(path)
    store = PGVector(
        embeddings=get_embeddings(provider),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", "documents"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    store.add_documents(documents=docs, ids=def_index_documents(docs))
    print(f"[ingest] Stored {len(docs)} chunks in collection '{collection}'.")

def get_documents_from_pdf(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    splits = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP, 
    add_start_index=False).split_documents(docs)
    if not splits:
        raise SystemExit("No document splits were created.")
    return enriched_documents(splits)

def enriched_documents(splits: list[Document]) -> list[Document]:
    return [
                Document(
                    page_content=doc.page_content,
                    metadata={
                        k: v for k, v in doc.metadata.items() if v not in ("", None)
                    },
                )
                for doc in splits
            ]

def def_index_documents(enriched: list[Document]) -> list[str]:
    return [f"doc-{i}" for i in range(len(enriched))]
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a PDF into Postgres pgvector via LangChain.")
    parser.add_argument("--pdf", help="Path to the PDF (default: PDF_PATH or ./document.pdf)")
    args = parser.parse_args()
    main(args.pdf)