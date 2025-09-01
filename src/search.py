from db import get_vectorstore

def similarity_search_with_score(query: str, k: int, provider: str):
    vs = get_vectorstore(provider)
    return vs.similarity_search_with_score(query, k=k)
