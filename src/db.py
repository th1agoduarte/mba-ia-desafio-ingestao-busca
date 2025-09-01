# src/db.py
import os
import psycopg
from typing import Optional
from langchain_postgres import PGVector
from common import normalize_for_psycopg, get_embeddings, validate_environment

# ------------------------
# Helpers de banco
# ------------------------
def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL não configurada")
    return psycopg.connect(normalize_for_psycopg(db_url))

def get_collection_id(conn: psycopg.Connection, collection_name: str) -> Optional[str]:
    row = conn.execute(
        "SELECT uuid FROM langchain_pg_collection WHERE name = %s LIMIT 1",
        (collection_name,),
    ).fetchone()
    return row[0] if row else None

def collection_has_any(conn: psycopg.Connection, collection_name: str) -> bool:
    cid = get_collection_id(conn, collection_name)
    if not cid:
        return False
    row = conn.execute(
        "SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s",
        (cid,),
    ).fetchone()
    return (row[0] or 0) > 0

def collection_has_source(conn: psycopg.Connection, collection_name: str, source_path: str) -> bool:
    cid = get_collection_id(conn, collection_name)
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

def delete_source(conn: psycopg.Connection, collection_name: str, source_path: str):
    cid = get_collection_id(conn, collection_name)
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

# ------------------------
# VectorStore
# ------------------------
def get_vectorstore(provider: str):
    required_vars = ["DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"]
    validate_environment(required_vars)
    embeddings = get_embeddings(provider)
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", "documents"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
