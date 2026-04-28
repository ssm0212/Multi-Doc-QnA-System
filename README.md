# 🔍 Multi-Document Question Answering System
### RAG Pipeline · Groq Free LLMs · FAISS Vector Store · Streamlit UI

A fully functional Retrieval-Augmented Generation (RAG) system that lets you upload multiple documents (PDF, DOCX, TXT) and ask questions about them. Powered entirely by **free APIs** — no OpenAI, no billing.

---

## 🏗️ Architecture

```
PDF / DOCX / TXT
       │
       ▼
[Document Loader]  →  chunks of text
       │
       ▼
[HuggingFace Embeddings]  ←  sentence-transformers/all-MiniLM-L6-v2  (FREE, runs locally)
       │
       ▼
[FAISS Vector Store]  ←  in-memory knowledge base
       │
    (on question)
       │
       ▼
[Question Embedding]  →  embed with same model
       │
       ▼
[Semantic Search]  →  top-K most relevant chunks
       │
       ▼
[Groq LLM API]  ←  Llama 3, Mixtral, Gemma  (FREE, 14,400 req/day)
       │
       ▼
[Answer + Source Citations]
```

---

## 🆓 LLMs Used (via Groq)

| Model | Speed | Notes |
|-------|-------|-------|
| `llama-3.3-70b-versatile` | Medium | ⭐ Best quality — recommended |
| `llama-3.1-8b-instant` | Fast | Great for simple Q&A |
| `mixtral-8x7b-32768` | Medium | Excellent reasoning |
| `gemma2-9b-it` | Fast | Google's balanced model |

All free via [Groq](https://console.groq.com) — 14,400 requests/day, no credit card needed.

---

## 🛠️ Tech Stack

| Component | Technology | Cost |
|-----------|------------|------|
| UI | Streamlit | Free |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free (local) |
| Vector DB | FAISS | Free (local) |
| LLM | Groq Inference API | Free |
| Orchestration | LangChain | Free |
| PDF parsing | PyPDF | Free |
| DOCX parsing | python-docx | Free |

---

## 📁 Project Structure

```
multi_doc_qa/
├── app.py                    # Streamlit UI — chat interface, sidebar, layout
├── requirements.txt          # Python dependencies
├── pyrightconfig.json        # VS Code Pylance config
├── .gitignore
├── README.md
├── .streamlit/
│   └── secrets.toml          # Local secrets — NOT pushed to GitHub
└── src/
    ├── document_loader.py    # PDF / DOCX / TXT loading and chunking
    ├── vector_store.py       # HuggingFace embeddings + FAISS index
    └── llm_engine.py         # Groq API + RAG pipeline
```

---

## ⚙️ Sidebar Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Chunk Size | Characters per text chunk | 500 |
| Chunk Overlap | Overlap between chunks | 50 |
| Top-K Retrieval | Chunks fed to LLM per question | 4 |
| Max Answer Tokens | LLM output length | 512 |

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Paste your **Groq API key** in the sidebar — get one free at [console.groq.com](https://console.groq.com)
2. Select a **free LLM** from the dropdown
3. Upload one or more **PDF, DOCX, or TXT** files
4. Click **⚡ Build Knowledge Base**
5. Type a question and get answers with source citations