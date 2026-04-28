"""
document_loader.py
Handles loading and chunking of multiple document types (PDF, DOCX, TXT).
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return list of Documents."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # Tag source
    for doc in docs:
        doc.metadata["source_file"] = Path(file_path).name
    return docs


def load_docx(file_path: str) -> List[Document]:
    """Load a DOCX file and return list of Documents."""
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = Path(file_path).name
    return docs


def load_txt(file_path: str) -> List[Document]:
    """Load a plain text file and return list of Documents."""
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = Path(file_path).name
    return docs


def load_document(file_path: str) -> List[Document]:
    """Route file to correct loader based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: Raw documents from loaders.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def load_and_chunk_files(file_paths: List[str], chunk_size=500, chunk_overlap=50) -> List[Document]:
    """
    Load and chunk multiple files.

    Returns:
        All chunks across all documents.
    """
    all_docs = []
    for fp in file_paths:
        try:
            docs = load_document(fp)
            all_docs.extend(docs)
        except Exception as e:
            print(f"[WARNING] Could not load {fp}: {e}")

    chunks = chunk_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks