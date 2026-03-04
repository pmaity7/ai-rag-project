# app.py — Run with: streamlit run app.py

import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

from utils.document_processor import load_pdf, load_txt, chunk_text
from utils.vector_store import build_vector_store
from utils.rag_chain import rag_answer

# ── Load API key from .env file (for local development) ───────────────────
load_dotenv()

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — Gemini RAG",
    page_icon="📄",
    layout="wide",
)

# ── Section 1: Session State Init ─────────────────────────────────────────
# These variables persist across Streamlit reruns

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Each item: {"user": "question", "assistant": "answer", "sources": [...]}

if "collection" not in st.session_state:
    st.session_state.collection = None     # ChromaDB collection object

if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None  # Authenticated Gemini client

if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False     # Whether a doc has been processed

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0       # Total chunks in current document

# ── Section 2: Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Setup")
    st.divider()

    # API Key input
    api_key = os.getenv("GEMINI_API_KEY", "")

    st.divider()

    # File uploader — only PDF and TXT supported
    uploaded_file = st.file_uploader(
        "📂 Upload Your Document",
        type=["pdf", "txt"],
        help="PDF or plain text files supported",
    )

    # Advanced settings — hidden by default to keep UI clean
    with st.expander("🛠️ Advanced Settings"):
        chunk_size = st.slider(
            "Chunk size (chars)", 200, 1000, 500, step=50,
            help="Larger = more context per chunk. Smaller = more precise retrieval."
        )
        chunk_overlap = st.slider(
            "Chunk overlap (chars)", 0, 200, 50, step=10,
            help="Overlap prevents losing context at chunk boundaries."
        )
        top_k = st.slider(
            "Chunks to retrieve (top-k)", 1, 8, 4,
            help="How many chunks to pass to Gemini per question."
        )

    st.divider()

    # Process button — triggers the full ingestion pipeline
    if st.button("🚀 Process Document", use_container_width=True, type="primary"):
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
        elif not uploaded_file:
            st.error("Please upload a document first.")
        else:
            with st.spinner("Reading, chunking and embedding your document..."):
                try:
                    # Step 1 — Load the file
                    if uploaded_file.name.endswith(".pdf"):
                        raw_text = load_pdf(uploaded_file)
                    else:
                        raw_text = load_txt(uploaded_file)

                    if not raw_text.strip():
                        st.error("Document appears empty or unreadable.")
                        st.stop()

                    # Step 2 — Chunk the text
                    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)

                    # Step 3 — Create Gemini client
                    client = genai.Client(api_key=api_key)

                    # Step 4 — Embed chunks and store in ChromaDB
                    collection = build_vector_store(chunks, client)

                    # Step 5 — Save everything to session state
                    st.session_state.collection = collection
                    st.session_state.gemini_client = client
                    st.session_state.doc_ready = True
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.chat_history = []  # Reset chat on new doc

                    st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

                except Exception as e:
                    st.error(f"Error processing document: {e}")

    # Show status if document is ready
    if st.session_state.doc_ready:
        st.info(f"📄 {st.session_state.chunk_count} chunks loaded and ready.")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Area ──────────────────────────────────────────────────────────────
st.title("📄 DocChat — Chat with Your Documents")
st.caption("Powered by Gemini 2.5 Flash + ChromaDB RAG")

# Onboarding screen — shown before any document is uploaded
if not st.session_state.doc_ready:
    st.info("👈 Upload a document and enter your Gemini API key in the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Add API Key")
        st.markdown("Paste your Gemini API key in the sidebar")
    with col2:
        st.markdown("### 2️⃣ Upload Document")
        st.markdown("PDF or .txt files up to ~50 pages work best")
    with col3:
        st.markdown("### 3️⃣ Start Chatting")
        st.markdown("Ask anything about your document")

# ── Section 3: Chat Interface ──────────────────────────────────────────────
else:
    # Replay all previous messages from session state
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["assistant"])
            # Section 4 — Show source chunks used for this answer
            with st.expander("📎 View Sources"):
                for i, source in enumerate(turn["sources"]):
                    st.markdown(f"**Excerpt {i + 1}:**")
                    st.caption(source)

    # Chat input box — appears at the bottom
    user_query = st.chat_input("Ask something about your document...")

    if user_query:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_query)

        # Get answer from RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = rag_answer(
                        query=user_query,
                        collection=st.session_state.collection,
                        client=st.session_state.gemini_client,
                        chat_history=st.session_state.chat_history,
                        top_k=top_k,
                    )
                    st.write(answer)

                    # Show which chunks were used
                    with st.expander("📎 View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Excerpt {i + 1}:**")
                            st.caption(source)

                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []
                    st.error(answer)

        # Save this turn to chat history
        st.session_state.chat_history.append({
            "user": user_query,
            "assistant": answer,
            "sources": sources,
        })