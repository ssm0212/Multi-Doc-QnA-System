# 🔍 Multi-Document Question Answering System
### RAG Pipeline · Free LLMs · FAISS Vector Store · Streamlit UI

---

## 📌 Architecture (mirrors the LangChain diagram)

```
PDF/DOCX/TXT
     │
     ▼
[Document Loader] ──► chunks of text
     │
     ▼
[HuggingFace Embeddings]  ←─ sentence-transformers/all-MiniLM-L6-v2 (FREE)
     │
     ▼
[FAISS Vector Store]  ←─ knowledge base / in-memory index
     │
  ┌──┘  (on user question)
  │
  ▼
[Question Embedding]  ──► embed question with same model
  │
  ▼
[Semantic Search]  ──► top-K most relevant chunks
  │
  ▼
[Free LLM via HF Inference API]  ←─ Mistral, Zephyr, Flan-T5, Falcon
  │
  ▼
[Answer + Source Citations]
```

---

## 🆓 Free LLMs Available (no billing needed)

| Model | Speed | Quality | HF ID |
|-------|-------|---------|-------|
| Flan-T5-Large | ⚡ Fast | Good for factual | `google/flan-t5-large` |
| Mistral-7B-Instruct | Medium | 🏆 Best quality | `mistralai/Mistral-7B-Instruct-v0.2` |
| Zephyr-7B-Beta | Medium | Great instruction | `HuggingFaceH4/zephyr-7b-beta` |
| Falcon-7B-Instruct | Medium | Solid open-source | `tiiuae/falcon-7b-instruct` |

---

## 🚀 Setup & Run

### 1. Clone / download this project

```bash
cd multi_doc_qa
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a FREE HuggingFace API Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a **free account** (no credit card needed)
3. Click **"New token"** → select **"Read"** role
4. Copy the token (starts with `hf_...`)

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 How to Use

1. **Paste your HuggingFace token** in the sidebar
2. **Select a free LLM** (start with Flan-T5-Large for speed)
3. **Upload documents** (PDF, DOCX, or TXT — multiple at once)
4. Click **"⚡ Build Knowledge Base"** — this embeds all chunks into FAISS
5. **Ask questions** in the chat — get answers with source citations!

---

## 📁 Project Structure

```
multi_doc_qa/
├── app.py                  # Streamlit UI
├── requirements.txt        # Python dependencies
├── README.md
└── src/
    ├── document_loader.py  # PDF/DOCX/TXT loading & chunking
    ├── vector_store.py     # HuggingFace embeddings + FAISS index
    └── llm_engine.py       # Free LLM integration + RAG chain
```

---

## 🔧 Configuration Options (in sidebar)

| Setting | Description | Default |
|---------|-------------|---------|
| Chunk Size | Characters per text chunk | 500 |
| Chunk Overlap | Overlap between chunks | 50 |
| Top-K Retrieval | How many chunks to feed LLM | 4 |
| Max Answer Tokens | LLM output length | 512 |

---

## 💡 Tips

- **First run**: The embedding model (~80MB) downloads automatically once, then is cached.
- **Large PDFs**: Use smaller chunk sizes (300–400) for more precise retrieval.
- **Multiple docs**: Works across all uploaded docs simultaneously.
- **Rate limits**: Free HF Inference API has rate limits. If you hit them, wait 1 min or use Flan-T5 (lighter).

---

## 🛠 Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| UI | Streamlit | Free |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free |
| Vector DB | FAISS (Facebook AI) | Free |
| LLM | HuggingFace Inference API | Free |
| Orchestration | LangChain | Free |
| Doc Parsing | PyPDF + python-docx | Free |
