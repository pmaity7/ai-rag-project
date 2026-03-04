# utils/rag_chain.py

from google import genai
from google.genai import types
from utils.vector_store import retrieve_relevant_chunks


def build_rag_prompt(
    query: str,
    context_chunks: list[str],
    chat_history: list[dict],
) -> str:
    """
    Constructs the full prompt sent to Gemini.

    The prompt has 4 clear sections:
      1. System instruction — tells Gemini its role and rules
      2. Document excerpts — the relevant chunks retrieved from ChromaDB
      3. Chat history — last 3 Q&A pairs for follow-up question support
      4. Current question — what the user just asked

    Args:
        query:          The user's current question
        context_chunks: Relevant chunks retrieved from ChromaDB
        chat_history:   List of previous {"user": ..., "assistant": ...} turns

    Returns:
        A fully constructed prompt string ready to send to Gemini
    """

    # Format each chunk with a numbered label for easy citation
    context_text = "\n\n---\n\n".join(
        [f"[Excerpt {i + 1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    # Include only the last 3 Q&A pairs to keep prompt size manageable
    history_text = ""
    if chat_history:
        history_text = "\n\nConversation so far:\n"
        for turn in chat_history[-6:]:  # -6 = last 3 pairs (user + assistant each)
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    prompt = f"""You are a helpful document assistant. Answer the user's question using ONLY the provided document excerpts below.

Rules:
- If the answer is clearly in the excerpts, answer directly and cite the excerpt (e.g. "According to Excerpt 2...")
- If the answer is NOT in the excerpts, say exactly: "I couldn't find that in the uploaded document."
- Never make up or assume information outside the excerpts.
- Be concise but complete.

--- DOCUMENT EXCERPTS ---
{context_text}
--- END OF EXCERPTS ---
{history_text}
Current Question: {query}

Answer:"""

    return prompt


def ask_gemini(prompt: str, client: genai.Client) -> str:
    """
    Sends the constructed prompt to Gemini 2.5 Flash and returns the response.

    temperature=0.2 keeps answers factual and consistent.
    max_output_tokens=1024 is enough for detailed answers without being excessive.

    Args:
        prompt: The fully constructed RAG prompt
        client: Authenticated Gemini client

    Returns:
        Gemini's response as a plain string
    """
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text


def rag_answer(
    query: str,
    collection,
    client: genai.Client,
    chat_history: list[dict],
    top_k: int = 4,
) -> tuple[str, list[str]]:
    """
    The full RAG pipeline in one function call.

    Flow:
      1. Retrieve top_k relevant chunks from ChromaDB for the query
      2. Build a grounded prompt using those chunks + chat history
      3. Send prompt to Gemini and get the answer back

    Args:
        query:        The user's current question
        collection:   ChromaDB collection from vector_store.py
        client:       Authenticated Gemini client
        chat_history: Previous Q&A turns for follow-up support
        top_k:        Number of chunks to retrieve (default 4)

    Returns:
        answer  (str):       Gemini's response
        sources (list[str]): The chunks used as context (shown in UI)
    """

    # Step 1 — Semantic search
    relevant_chunks = retrieve_relevant_chunks(query, collection, client, top_k)

    # Step 2 — Build prompt
    prompt = build_rag_prompt(query, relevant_chunks, chat_history)

    # Step 3 — Get answer from Gemini
    answer = ask_gemini(prompt, client)

    return answer, relevant_chunks