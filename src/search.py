import os
from langchain_postgres import PGVector
from common import get_embeddings, validate_environment

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

def similarity_search_with_score(query: str, k: int, provider: str):
    vs = get_vectorstore(provider)
    return vs.similarity_search_with_score(query, k=k)
