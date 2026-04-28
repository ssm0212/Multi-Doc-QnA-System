"""
vector_store.py
Manages embeddings (via free HuggingFace sentence-transformers) and FAISS vector store.
"""

import os
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Model config ────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is small (~80MB), fast, and completely free.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embeddings object (downloaded once, cached locally)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document], persist_dir: Optional[str] = None) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Args:
        chunks: Chunked documents to embed.
        persist_dir: If provided, save the index to this directory.

    Returns:
        FAISS vector store ready for similarity search.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore.save_local(persist_dir)

    return vectorstore


def load_vectorstore(persist_dir: str) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


def add_documents_to_store(vectorstore: FAISS, new_chunks: List[Document]) -> FAISS:
    """Add new document chunks to an existing FAISS store."""
    embeddings = get_embeddings()
    vectorstore.add_documents(new_chunks)
    return vectorstore


def semantic_search(vectorstore: FAISS, query: str, k: int = 4) -> List[Document]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        vectorstore: The FAISS index to search.
        query: User's question.
        k: Number of results to retrieve.

    Returns:
        List of top-k Document chunks with scores.
    """
    results = vectorstore.similarity_search(query, k=k)
    return results
