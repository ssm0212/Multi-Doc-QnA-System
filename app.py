"""
app.py
Streamlit UI for the Multi-Document Question Answering System (RAG + Free LLMs)

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import streamlit as st

# Allow src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from document_loader import load_and_chunk_files
from vector_store import build_vectorstore, semantic_search
from llm_engine import FREE_MODELS, semantic_search_and_answer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Doc QA System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.main-header h1 {
    font-family: 'Space Mono', monospace;
    color: #00f5d4;
    font-size: 1.8rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: rgba(255,255,255,0.6);
    margin: 0.4rem 0 0 0;
    font-size: 0.9rem;
}

.pipeline-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.step-num {
    background: #00f5d4;
    color: #0f0c29;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.75rem;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.answer-box {
    background: linear-gradient(135deg, rgba(0,245,212,0.05), rgba(48,43,99,0.3));
    border: 1px solid rgba(0,245,212,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #00f5d4;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

.badge {
    background: rgba(0,245,212,0.15);
    color: #00f5d4;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #00f5d4, #00b4d8) !important;
    color: #0f0c29 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Multi-Document QA System</h1>
    <p>RAG Pipeline · Free LLMs · PDF / DOCX / TXT · Semantic Search · FAISS Vector Store</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    hf_token = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
        help="Free token from huggingface.co/settings/tokens",
    )
    st.caption("🔗 Get a free key at [console.groq.com](https://console.groq.com/keys)")

    st.markdown("---")
    st.markdown("### 🤖 Select Free LLM")
    model_label = st.selectbox("Model", list(FREE_MODELS.keys())) or list(FREE_MODELS.keys())[0]
    selected_model = FREE_MODELS[model_label]
    st.caption(f"`{selected_model}`")

    st.markdown("---")
    st.markdown("### 🔧 RAG Settings")
    chunk_size = st.slider("Chunk Size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 200, 50, 10)
    top_k = st.slider("Top-K Chunks to Retrieve", 1, 8, 4)
    max_tokens = st.slider("Max Answer Tokens", 128, 1024, 512, 64)

    st.markdown("---")
    st.markdown("### 📋 RAG Pipeline")
    steps = [
        ("1", "📄 Upload documents"),
        ("2", "✂️ Chunk text"),
        ("3", "🔢 Embed chunks → FAISS"),
        ("4", "❓ Embed question"),
        ("5", "🔍 Semantic search"),
        ("6", "🤖 LLM generates answer"),
    ]
    for num, label in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-num">{num}</div>
            <span style="color:rgba(255,255,255,0.8);font-size:0.85rem">{label}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Main columns ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.4], gap="large")

# ── LEFT: Upload & Index ──────────────────────────────────────────────────────
with col1:
    st.markdown("### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f'<span class="badge">📄 {f.name}</span> <small style="color:rgba(255,255,255,0.4)">{size_kb:.1f} KB</small>', unsafe_allow_html=True)

    build_btn = st.button("⚡ Build Knowledge Base", use_container_width=True, disabled=not uploaded_files)

    if build_btn and uploaded_files:
        with st.spinner("📑 Loading & chunking documents..."):
            # Save to temp files
            tmp_paths = []
            for uf in uploaded_files:
                suffix = os.path.splitext(uf.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uf.getvalue())
                tmp.close()
                tmp_paths.append(tmp.name)

            chunks = load_and_chunk_files(tmp_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        with st.spinner(f"🔢 Embedding {len(chunks)} chunks → FAISS (downloading model first time)..."):
            vectorstore = build_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_names = [f.name for f in uploaded_files]

        # Cleanup tmp files
        for p in tmp_paths:
            os.unlink(p)

        st.success(f"✅ Knowledge base built! {len(chunks)} chunks indexed from {len(uploaded_files)} document(s).")

    # Show current state
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("**📚 Indexed Documents:**")
        for name in st.session_state.doc_names:
            st.markdown(f"- `{name}`")

# ── RIGHT: Q&A Interface ──────────────────────────────────────────────────────
with col2:
    st.markdown("### 💬 Ask a Question")

    # Scrollable chat history — always above the input box
    chat_container = st.container(height=520)
    with chat_container:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
                if entry.get("sources"):
                    with st.expander(f"📎 {len(entry['sources'])} source chunk(s) retrieved"):
                        for src in entry["sources"]:
                            fname = src.metadata.get("source_file", "Unknown")
                            page = src.metadata.get("page", "")
                            page_str = f" · Page {page+1}" if page != "" else ""
                            st.markdown(f"""
                            <div class="source-card">
                                <strong style="color:#00f5d4">📄 {fname}{page_str}</strong><br>
                                <span style="color:rgba(255,255,255,0.7)">{src.page_content[:300]}{'...' if len(src.page_content) > 300 else ''}</span>
                            </div>
                            """, unsafe_allow_html=True)

    # Input pinned below the chat history
    question = st.chat_input("Ask anything about your documents...")

    if question:
        if not st.session_state.vectorstore:
            st.warning("⚠️ Please upload and build the knowledge base first.")
        elif not hf_token:
            st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
        else:
            with st.spinner(f"🔍 Searching docs → 🤖 Generating answer with {model_label}..."):
                try:
                    result = semantic_search_and_answer(
                        vectorstore=st.session_state.vectorstore,
                        question=question,
                        hf_token=hf_token,
                        model_id=selected_model,
                        k=top_k,
                        max_new_tokens=max_tokens,
                    )
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["source_documents"],
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}\n\nCheck your Groq API key and model availability.")
            st.rerun()

    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()