# 📄 DocChat — Chat with Your Documents

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload a PDF or text file and ask questions about it. Powered by **Gemini 2.5 Flash** for answers and **gemini-embedding-001** for semantic search. Built with Streamlit and ChromaDB.

---

## 🧠 How It Works

```
User uploads PDF/TXT
        ↓
document_processor.py  →  loads and splits text into chunks
        ↓
vector_store.py        →  embeds chunks via Gemini + stores in ChromaDB
        ↓
User asks a question
        ↓
vector_store.py        →  embeds query + retrieves top-4 relevant chunks
        ↓
rag_chain.py           →  builds grounded prompt + calls Gemini 2.5 Flash
        ↓
app.py                 →  displays answer + shows source excerpts
```

The model **only answers from your document**. If the answer isn't in the document, it says so — no hallucinations.

---

## 🗂️ Project Structure

```
rag-gemini/
├── app.py                        # Streamlit UI — main entry point
├── Dockerfile                    # Container recipe
├── requirements.txt              # Python dependencies
├── .env                          # Your API key (never commit this)
├── .env.example                  # Template for others to copy
├── .gitignore
├── .dockerignore
└── utils/
    ├── __init__.py
    ├── document_processor.py     # Load PDF/TXT + chunk text
    ├── vector_store.py           # Embed chunks + store/query ChromaDB
    └── rag_chain.py              # Build prompt + call Gemini
```

---

## 🆓 Tech Stack

| Tool | Purpose | Cost |
|---|---|---|
| [Gemini 2.5 Flash](https://ai.google.dev) | LLM for answering questions | Free tier / Pay as you go |
| [gemini-embedding-001](https://ai.google.dev/gemini-api/docs/embeddings) | Text embeddings | Free tier / Pay as you go |
| [ChromaDB](https://docs.trychroma.com) | In-memory vector database | Free / Open source |
| [Streamlit](https://streamlit.io) | Web UI | Free / Open source |
| [pypdf](https://pypdf.readthedocs.io) | PDF text extraction | Free / Open source |
| [LangChain](https://python.langchain.com) | Text chunking | Free / Open source |
| [Docker](https://docker.com) | Containerization | Free |

---

## ⚙️ Prerequisites

- Python 3.11 (important — newer versions have library compatibility issues)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Docker Desktop (only if running via Docker)

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-gemini.git
cd rag-gemini
```

### 2. Create and activate a virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Running with Docker

### 1. Build the image
```bash
docker build -t rag-gemini .
```

### 2. Run the container
```bash
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key_here rag-gemini
```

Open `http://localhost:8501` in your browser.

> Your `.env` file is excluded from the Docker image via `.dockerignore`. Always pass the API key using the `-e` flag at runtime.

---

## 💬 Usage

1. Open the app in your browser
2. Upload a PDF or `.txt` file using the sidebar
3. Click **Process Document** and wait for the chunks to be indexed
4. Type your question in the chat box
5. The app answers using only the content from your document
6. Expand **View Sources** under any answer to see which excerpts were used

---

## ⚙️ Advanced Settings

Available in the sidebar under **Advanced Settings**:

| Setting | Default | Description |
|---|---|---|
| Chunk size | 500 chars | Max size of each text chunk |
| Chunk overlap | 50 chars | Overlap between adjacent chunks to preserve context |
| Top-k retrieval | 4 | Number of chunks passed to Gemini per question |

---

## 📁 File Size Recommendations

| File Size | Expected Behaviour |
|---|---|
| < 1MB (1–10 pages) | Processes in seconds, ideal for testing |
| 1–10MB (10–100 pages) | Takes 1–2 minutes to embed |
| > 10MB | May hit API rate limits, not recommended |

---

## ⚠️ Known Limitations

- ChromaDB runs **in-memory** — all indexed data is lost when the app restarts. Upload your document again after each restart.
- Only **PDF and plain text** files are supported. Scanned PDFs with no extractable text will not work.
- Requires **Python 3.11** specifically due to ChromaDB and Pydantic compatibility.

---

## 🔭 What's Next

- [ ] Support multiple documents in one session
- [ ] Add page number citations to source excerpts
- [ ] Persistent ChromaDB storage across sessions
- [ ] Deploy on Hugging Face Spaces

---

## 📄 License

MIT
