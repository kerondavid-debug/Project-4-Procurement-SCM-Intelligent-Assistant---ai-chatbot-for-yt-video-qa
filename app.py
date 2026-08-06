import streamlit as st
import json
import chromadb
from openai import OpenAI

st.set_page_config(page_title="Procurement Intelligence Assistant", page_icon="📦")

# --- Setup (cached so this only runs once per session) ---
@st.cache_resource
def init_clients():
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="procurement_scm",
        metadata={"hnsw:space": "cosine"}
    )

    # Build the index if it's empty (first run / fresh deploy)
    if collection.count() == 0:
        with open("data/chunks.json") as f:
            chunks = json.load(f)

        def embed_batch(texts):
            r = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in r.data]

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [f"{c['video_id']}_{i+j}" for j, c in enumerate(batch)]
            texts = [c["text"] for c in batch]
            embeddings = embed_batch(texts)
            metadatas = [
                {"video_id": c["video_id"], "title": c["title"], "start": c["start"], "end": c["end"]}
                for c in batch
            ]
            collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return client, collection
client, collection = init_clients()

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def answer_question_with_memory(question: str, history: list, n_results: int = 4) -> dict:
    results = collection.query(
        query_embeddings=[embed_text(question)],
        n_results=n_results
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {m['title']} at {m['start']:.0f}s]\n{d}"
        for d, m in zip(docs, metas)
    )

    system_prompt = (
        "You are a procurement and supply chain management assistant. "
        "Answer using ONLY the provided context. If the context is "
        "insufficient, say so. Cite the video title when referencing "
        "specific information. Use the conversation history to "
        "understand follow-up questions."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"title": m["title"], "start": m["start"], "video_id": m["video_id"]}
            for m in metas
        ]
    }


# --- UI ---
st.title("📦 Procurement Intelligence Assistant")
st.caption("Ask questions about procurement, sourcing, and supply chain management — answers are grounded in a curated set of expert video content.")

if "messages" not in st.session_state:
    st.session_state.messages = []       # for display
    st.session_state.llm_history = []    # for the model (Q&A only, no bulky context)

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- [{s['title']} ({s['start']:.0f}s)]"
                        f"(https://youtube.com/watch?v={s['video_id']}&t={int(s['start'])})"
                    )

# Chat input
if question := st.chat_input("Ask a procurement or supply chain question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question_with_memory(question, st.session_state.llm_history)
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"- [{s['title']} ({s['start']:.0f}s)]"
                        f"(https://youtube.com/watch?v={s['video_id']}&t={int(s['start'])})"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    st.session_state.llm_history.append({"role": "user", "content": question})
    st.session_state.llm_history.append({"role": "assistant", "content": result["answer"]})