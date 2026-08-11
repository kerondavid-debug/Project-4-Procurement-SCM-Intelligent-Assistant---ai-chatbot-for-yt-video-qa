import json
import os
import traceback

import streamlit as st
import chromadb
from openai import OpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Procurement Intelligence Assistant", page_icon="📦")

# Optional: LangSmith tracing (only activates if these are set in the deploy
# environment / .env — never overwrite a value the user has already set).
os.environ.setdefault("LANGCHAIN_PROJECT", "procurement-scm-assistant")


# --------------------------------------------------------------------------
# Setup (cached so this only runs once per process, not once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def init_clients():
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="procurement_scm",
        metadata={"hnsw:space": "cosine"},
    )

    with open("data/chunks.json") as f:
        chunks = json.load(f)

    # Build the index if it's empty (first run / fresh deploy). Visible so
    # that if a tester lands here before the warm-up run, they see progress
    # instead of a silent stall.
    if collection.count() == 0:
        status = st.info("First run: building the video index, this takes a minute...")

        def embed_batch(texts):
            r = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in r.data]

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [f"{c['video_id']}_{i + j}" for j, c in enumerate(batch)]
            texts = [c["text"] for c in batch]
            embeddings = embed_batch(texts)
            metadatas = [
                {"video_id": c["video_id"], "title": c["title"], "start": c["start"], "end": c["end"]}
                for c in batch
            ]
            collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        status.empty()
    elif collection.count() != len(chunks):
        # A prebuilt chroma_db is shipped in this deploy (per our chunks.json
        # / chroma_db sync check). If the counts ever drift — chunks.json
        # updated without rebuilding the index, or vice versa — surface it
        # instead of silently serving a stale/partial index.
        st.warning(
            f"Index/data mismatch: chroma_db has {collection.count()} chunks "
            f"indexed but data/chunks.json has {len(chunks)}. Rebuild the "
            "index before relying on results."
        )

    # Video list for the metadata tool, derived from chunks so deployment
    # doesn't need the raw transcripts.json bundled alongside it.
    video_index = {}
    for c in chunks:
        video_index.setdefault(c["video_id"], c["title"])

    return client, collection, video_index


@st.cache_resource
def build_agent(_client, _collection, _video_index):
    """Build the tool-calling agent once per process. Leading underscores on
    the params tell st.cache_resource not to hash them."""

    def embed_text(text: str) -> list[float]:
        response = _client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    # --- Tool 1: video content search (RAG over transcript chunks) ---
def retrieve_video_content(query: str) -> str:
    results = _collection.query(query_embeddings=[embed_text(query)], n_results=4)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Stash structured sources for the UI to render after the agent finishes —
    # the agent only sees/returns the text below, not this list.
    st.session_state.last_sources = metas

    return "\n\n".join(
        f"[Source: {m['title']} at {m['start']:.0f}s]\n{d}" for d, m in zip(docs, metas)
    )

    video_retrieval_tool = Tool(
        name="video_content_search",
        func=retrieve_video_content,
        description=(
            "Searches a curated set of procurement and supply chain management "
            "video transcripts. Use this for any question about procurement, "
            "sourcing, contracts, inventory, demand planning, logistics, risk "
            "management, supplier relationships, sustainability, or AI in "
            "procurement. Returns relevant excerpts with source citations."
        ),
    )

    # --- Tool 2: general knowledge fallback (no retrieval) ---
    def answer_general_knowledge(query: str) -> str:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a procurement and supply chain management "
                        "assistant. Answer using your general knowledge of "
                        "the field. Be clear and concise. Note at the start "
                        "of your answer that this is general knowledge, not "
                        "sourced from the video library."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    general_knowledge_tool = Tool(
        name="general_procurement_knowledge",
        func=answer_general_knowledge,
        description=(
            "Answers general procurement, sourcing, or supply chain "
            "management questions using broad domain knowledge, when the "
            "question is not covered by the indexed video library — for "
            "example, definitions, industry standards, certifications, or "
            "topics the videos don't address. Does not provide source "
            "citations from the videos."
        ),
    )

    # --- Tool 3: video library metadata (no semantic search) ---
    def get_video_metadata(query: str) -> str:
        video_list = [f"- {title} (video_id: {vid})" for vid, title in _video_index.items()]
        return f"The library contains {len(_video_index)} videos:\n" + "\n".join(video_list)

    video_metadata_tool = Tool(
        name="video_library_metadata",
        func=get_video_metadata,
        description=(
            "Answers questions about the video library's contents as a "
            "whole — for example, how many videos are indexed, what topics "
            "or titles are covered, or which video discusses a specific "
            "subject at a high level. Does NOT search inside video "
            "transcripts — use video_content_search for that instead."
        ),
    )

    tools = [video_retrieval_tool, general_knowledge_tool, video_metadata_tool]

    agent_system_prompt = (
        "You are a procurement and supply chain management assistant. "
        "You have three tools available:\n"
        "- video_content_search: searches inside the curated video library "
        "for specific procurement/SCM content\n"
        "- video_library_metadata: answers questions about the library "
        "itself — titles, topic coverage, video count\n"
        "- general_procurement_knowledge: general domain knowledge, not "
        "sourced from videos\n\n"
        "Use video_library_metadata for questions about what the library "
        "contains or covers, not for questions seeking specific content "
        "from within a video. For content questions, MANDATORY RULE: call "
        "video_content_search FIRST. Only use general_procurement_knowledge "
        "if video_content_search's results are empty, irrelevant, or "
        "insufficient."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


def transcribe_audio(client: OpenAI, audio_file) -> str:
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text


def render_sources(sources):
    """Single source-of-truth for source-link rendering, so the historical
    and freshly-generated messages never drift out of sync (app.py used to
    have two copies of this with a broken markdown link in one of them)."""
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"- [{s['title']} ({s['start']:.0f}s)]"
                f"(https://www.youtube.com/watch?v={s['video_id']}&t={int(s['start'])})"
            )


# --------------------------------------------------------------------------
# App setup / state
# --------------------------------------------------------------------------
try:
    client, collection, video_index = init_clients()
    agent_executor = build_agent(client, collection, video_index)
except Exception as e:
    st.error(f"Failed to initialize the assistant: {e}")
    st.stop()

# --- TEMPORARY DEBUG CHECK ---
if agent_executor is None:
    st.error("agent_executor is None right after build_agent() — check build_agent's return path.")
    st.stop()
else:
    st.write(f"DEBUG: agent_executor type = {type(agent_executor)}")
# --- END DEBUG ---


st.title("📦 Procurement_SCM_Intelligence Assistant")
st.caption(
    "Ask questions about procurement, sourcing, and supply chain management — "
    "answers are grounded in a curated set of expert video content, with a "
    "general-knowledge fallback for questions outside the library."
)

if "messages" not in st.session_state:
    st.session_state.messages = []          # for display
    st.session_state.chat_history = []      # LangChain messages, for agent memory
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None


def handle_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.session_state.last_sources = []  # reset before each query
            try:
                result = agent_executor.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
                answer = result["output"]
            except Exception as e:
                traceback.print_exc()
                answer = f"Sorry, something went wrong answering that: {e}"

            st.markdown(answer)
            sources = st.session_state.last_sources
            if sources:
                render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
   

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Voice input
audio_file = st.file_uploader("Or ask by voice 🎤", type=["wav", "mp3", "m4a"])
voice_question = None
if audio_file is not None and audio_file.file_id != st.session_state.last_audio_id:
    # Only transcribe/submit a given upload once — otherwise every rerun
    # (e.g. after a normal text question) would resubmit the same audio
    # question as long as the file stays in the uploader widget.
    st.session_state.last_audio_id = audio_file.file_id
    with st.spinner("Transcribing..."):
        try:
            voice_question = transcribe_audio(client, audio_file)
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")

# Chat input
question = st.chat_input("Ask a procurement or supply chain question...")
if question:
    handle_question(question)
elif voice_question:
    handle_question(voice_question)
