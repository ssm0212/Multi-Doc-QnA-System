"""
llm_engine.py
Uses Groq's free API — fastest free LLM inference available.
Free tier: 14,400 requests/day, no credit card needed.

Sign up at: https://console.groq.com  (free, takes 30 seconds)
Get API key: https://console.groq.com/keys
"""

from langchain_community.vectorstores import FAISS

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""


FREE_MODELS = {
    "Llama-3.3-70B (Best quality ⭐)": "llama-3.3-70b-versatile",
    "Llama-3.1-8B (Fast)": "llama-3.1-8b-instant",
    "Mixtral-8x7B (Great reasoning)": "mixtral-8x7b-32768",
    "Gemma2-9B (Google, balanced)": "gemma2-9b-it",
}


def call_groq(
    model_id: str,
    api_key: str,
    prompt: str,
    max_tokens: int = 512,
) -> str:
    """Call Groq's free API — OpenAI-compatible, extremely fast."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


def semantic_search_and_answer(
    vectorstore: FAISS,
    question: str,
    hf_token: str,       # kept as param name for compatibility — now holds Groq key
    model_id: str,
    k: int = 4,
    max_new_tokens: int = 512,
) -> dict:
    source_docs = vectorstore.similarity_search(question, k=k)

    context_parts = []
    for i, doc in enumerate(source_docs):
        fname = doc.metadata.get("source_file", "document")
        context_parts.append(f"[Source {i+1}: {fname}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    answer = call_groq(model_id, hf_token, prompt, max_new_tokens)

    return {
        "answer": answer,
        "source_documents": source_docs,
    }