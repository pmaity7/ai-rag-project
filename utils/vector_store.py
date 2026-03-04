# utils/vector_store.py

import chromadb
from google import genai
from google.genai import types


def get_chroma_client():
    """
    Creates and returns an in-memory ChromaDB client.

    In-memory means all data lives in RAM and resets when
    the app restarts. Perfect for development — no setup needed.
    For production you'd switch to persistent storage.
    """
    return chromadb.Client()


def embed_text(text: str, client: genai.Client) -> list[float]:
    """
    Converts a document chunk into a 3072-dimensional vector.
    Uses task_type RETRIEVAL_DOCUMENT — tells Gemini this is
    content being stored, not a search query.
    """
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def embed_query(query: str, client: genai.Client) -> list[float]:
    """
    Converts a user question into a 3072-dimensional vector.
    Uses task_type RETRIEVAL_QUERY — tells Gemini this is
    a search query, not a document being stored.
    """
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def build_vector_store(chunks: list[str], client: genai.Client, collection_name: str = "docs"):
    """
    Embeds all chunks and stores them in ChromaDB.

    Flow:
      1. Create a fresh ChromaDB collection (wipes old one if exists)
      2. Loop through each chunk and embed it via Gemini
      3. Batch insert all chunks + their vectors into ChromaDB

    Args:
        chunks:          List of text chunks from document_processor.py
        client:          Authenticated Gemini client
        collection_name: Name of the ChromaDB collection

    Returns:
        The ChromaDB collection object (used later for retrieval)
    """
    chroma_client = get_chroma_client()

    # Always start fresh on a new document upload
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for semantic search
    )

    print(f"Embedding {len(chunks)} chunks... this may take a moment.")

    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk, client)
        embeddings.append(embedding)
        print(f"  Embedded chunk {i + 1}/{len(chunks)}")

    # Insert everything into ChromaDB in one batch
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"✅ {len(chunks)} chunks stored in ChromaDB.")
    return collection


def retrieve_relevant_chunks(
    query: str,
    collection,
    client: genai.Client,
    top_k: int = 4,
) -> list[str]:
    """
    Finds the top_k most relevant chunks for a given query.

    Flow:
      1. Embed the user's query
      2. ChromaDB compares it against all stored chunk vectors
      3. Returns the top_k closest matches by cosine similarity

    Args:
        query:      The user's question
        collection: ChromaDB collection from build_vector_store()
        client:     Authenticated Gemini client
        top_k:      Number of chunks to retrieve (4 is a good default)

    Returns:
        List of the most relevant text chunks
    """
    query_embedding = embed_query(query, client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # results["documents"] is a list of lists (one per query)
    return results["documents"][0]