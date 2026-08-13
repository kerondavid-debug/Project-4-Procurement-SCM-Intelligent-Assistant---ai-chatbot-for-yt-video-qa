import json
import os
import re
import threading
import time
import traceback

from dotenv import load_dotenv
load_dotenv()  # loads .env for local runs — Streamlit Cloud injects its
# own Secrets into the environment automatically when deployed, so this
# is a no-op there, but local `streamlit run app.py` needs it explicitly.

import streamlit as st
import chromadb
from openai import OpenAI

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # hybrid search degrades to embeddings-only if not installed

try:
    import cohere
except ImportError:
    cohere = None  # reranking is skipped if not installed / no API key

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

    return client, collection, video_index, chunks


@st.cache_resource
def build_agent(_client, _collection, _video_index, _chunks):
    """Build the tool-calling agent once per process. Leading underscores on
    the params tell st.cache_resource not to hash them."""

    def embed_text(text: str) -> list[float]:
        response = _client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    # --- Tool 1: video content search (hybrid retrieval + rerank) ---
    RELEVANCE_THRESHOLD = 0.45  # cosine distance fallback, used only when
    # Cohere reranking isn't configured (no COHERE_API_KEY) — same
    # embeddings-only filtering as before.
    RERANK_THRESHOLD = 0.3      # Cohere relevance score (0-1) when reranking
    # is available. Tune based on testing; 0.3 is a permissive default.
    EMBED_TOP_K = 15
    BM25_TOP_K = 15
    FINAL_TOP_K = 6

    # Cohere Trial keys are capped at 10 API calls/minute. A single
    # interactive user is unlikely to hit that, but a retry-with-backoff
    # costs nothing when unused and prevents a stray 429 from crashing a
    # question outright (confirmed happening under concurrent load in
    # langsmith_eval.py — same rerank endpoint, same limit).
    _cohere_rate_lock = threading.Lock()
    _cohere_last_call_time = [0.0]
    _COHERE_MIN_INTERVAL = 6.5

    def _rerank_with_rate_limit(cohere_client, **kwargs):
        max_retries = 4
        for attempt in range(max_retries):
            with _cohere_rate_lock:
                wait = _COHERE_MIN_INTERVAL - (time.time() - _cohere_last_call_time[0])
                if wait > 0:
                    time.sleep(wait)
                try:
                    result = cohere_client.rerank(**kwargs)
                    _cohere_last_call_time[0] = time.time()
                    return result
                except Exception as e:
                    _cohere_last_call_time[0] = time.time()
                    is_rate_limit = "429" in str(e) or "TooManyRequests" in type(e).__name__
                    if is_rate_limit and attempt < max_retries - 1:
                        time.sleep(15)
                        continue
                    raise

    # BM25 index over the same chunks Chroma was built from. Pure embedding
    # search can miss exact-phrase / proper-noun matches (e.g. a tool name
    # like "Icertis") that a keyword-based search catches directly — hybrid
    # search combines both instead of relying on embeddings alone.
    _bm25 = None
    if BM25Okapi is not None and _chunks:
        _bm25_tokenized = [re.findall(r"\w+", c["text"].lower()) for c in _chunks]
        _bm25 = BM25Okapi(_bm25_tokenized)

    _cohere_client = None
    if cohere is not None and os.environ.get("COHERE_API_KEY"):
        _cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])

    def _format_chunks(docs, metas):
        return "\n\n".join(
            f"[Source: {m['title']} | video_id: {m['video_id']} at {m['start']:.0f}s]\n{d}"
            for d, m in zip(docs, metas)
        )

    def retrieve_video_content(query: str) -> str:
        results = _collection.query(
            query_embeddings=[embed_text(query)],
            n_results=EMBED_TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        emb_docs = results["documents"][0]
        emb_metas = results["metadatas"][0]
        emb_dists = results["distances"][0]

        # No hybrid/rerank available (packages or API key missing) — fall
        # back to the original embeddings-only behavior unchanged.
        if _bm25 is None or _cohere_client is None:
            relevant = [
                (d, m) for d, m, dist in zip(emb_docs, emb_metas, emb_dists)
                if dist <= RELEVANCE_THRESHOLD
            ]
            if not relevant:
                st.session_state.last_sources = []
                return (
                    "No sufficiently relevant content found in the video "
                    "library for this query. Use general_procurement_"
                    "knowledge instead."
                )
            docs, metas = zip(*relevant[:FINAL_TOP_K])
            st.session_state.last_sources = list(metas)
            return _format_chunks(docs, metas)

        # Hybrid: merge embedding candidates with BM25 candidates, dedupe
        # by (video_id, start), then rerank the merged set.
        tokenized_query = re.findall(r"\w+", query.lower())
        bm25_scores = _bm25.get_scores(tokenized_query)
        top_bm25_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:BM25_TOP_K]

        candidates = {}
        for d, m in zip(emb_docs, emb_metas):
            candidates[(m["video_id"], round(m["start"], 1))] = (d, m)
        for i in top_bm25_idx:
            c = _chunks[i]
            key = (c["video_id"], round(c["start"], 1))
            candidates.setdefault(
                key,
                (c["text"], {"title": c["title"], "video_id": c["video_id"], "start": c["start"]}),
            )

        candidate_list = list(candidates.values())
        if not candidate_list:
            st.session_state.last_sources = []
            return (
                "No sufficiently relevant content found in the video "
                "library for this query. Use general_procurement_"
                "knowledge instead."
            )

        docs_text = [d for d, _ in candidate_list]
        rerank_resp = _rerank_with_rate_limit(
            _cohere_client,
            model="rerank-english-v3.0",
            query=query,
            documents=docs_text,
            top_n=min(FINAL_TOP_K, len(docs_text)),
        )
        relevant = [
            candidate_list[r.index]
            for r in rerank_resp.results
            if r.relevance_score >= RERANK_THRESHOLD
        ]

        if not relevant:
            st.session_state.last_sources = []
            return (
                "No sufficiently relevant content found in the video "
                "library for this query. Use general_procurement_"
                "knowledge instead."
            )

        docs, metas = zip(*relevant)
        st.session_state.last_sources = list(metas)
        return _format_chunks(docs, metas)

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
                        "the field. Be clear, concise, and write in plain "
                        "prose rather than a numbered/bulleted breakdown "
                        "unless the question specifically asks for a list — "
                        "this keeps general-knowledge answers visually "
                        "distinct from video-sourced ones, which use "
                        "structured citations. Do not add your own "
                        "disclaimer about the source; the app adds that "
                        "label separately. If the question is entirely "
                        "unrelated to procurement or supply chain (e.g. "
                        "weather, recipes, general trivia), do not reframe "
                        "it into a procurement angle or partially answer "
                        "it — state plainly that it's outside scope and "
                        "invite a relevant question instead."
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
        "For most questions, a video library search has ALREADY been run "
        "for you and the results are included at the top of the human "
        "message, labeled 'Video library search results'. Use them "
        "directly if they answer the question — you do NOT need to call "
        "video_content_search again for the same question. Only call it "
        "yourself if you need a meaningfully different or more specific "
        "search.\n\n"
        "Routing:\n"
        "1. If the question asks what the library contains, covers, or how "
        "many videos it has, call video_library_metadata.\n"
        "2. If the provided search results say 'No sufficiently relevant "
        "content found', or the question asks something those results "
        "don't cover — including general definitions, standards, or "
        "certifications (e.g. CIPS) — call general_procurement_knowledge "
        "for that part.\n"
        "3. A single question can need more than one tool — e.g. one part "
        "answerable from the provided video results and another part that "
        "isn't. Call whatever combination of tools is needed to fully "
        "answer every part of the question; don't drop a part just "
        "because another part was already covered.\n"
        "4. If a question is entirely outside procurement/supply chain, "
        "call general_procurement_knowledge and let it explain that the "
        "topic is out of scope.\n"
        "5. When you use the video search results — whether pre-provided "
        "or from your own video_content_search call — answer using ONLY "
        "what those excerpts actually state. Do not add examples, tools, "
        "steps, or facts from your own general knowledge, even ones that "
        "seem obviously true or fit naturally alongside what's retrieved "
        "— a plausible addition is not something you actually know is in "
        "THIS video. If the excerpts don't fully answer the question, say "
        "so explicitly rather than filling the gap yourself; call "
        "general_procurement_knowledge for that instead, and make clear "
        "in your answer which parts came from the video versus general "
        "knowledge."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    # --- Retrieve-first wrapper ---
    # Root cause of a recurring eval failure: on questions that sound like a
    # "well-known" business framework (e.g. "6 steps in the strategic sourcing
    # flywheel"), gpt-4o-mini sometimes skips calling video_content_search
    # entirely on its first turn and answers from its own training data
    # instead — even with an explicit "CRITICAL RULE: call
    # video_content_search FIRST" in the system prompt. A stronger prompt
    # doesn't reliably fix a model discretion problem, so retrieval is no
    # longer optional: it always runs first, in code, before the agent gets
    # a chance to decide whether to bother.
    #
    # First version of this fix (bypassed the agent entirely and answered
    # directly whenever retrieval found something) fixed that bug but broke
    # compound questions — e.g. "what does the contract video cover, AND
    # what's the CIPS definition of ethical sourcing" would answer only the
    # video half and silently drop the CIPS half, since bypassing the agent
    # meant general_procurement_knowledge never got a chance to run. This
    # version instead always runs the agent, but with retrieval results
    # already attached to its input — so the agent can't skip searching
    # (already done), but can still reach for a second tool when a question
    # has a part the video results don't cover.
    def answer_question(question: str, chat_history: list) -> dict:
        retrieved = retrieve_video_content(question)
        augmented_input = (
            f"Video library search results for this question:\n{retrieved}\n\n"
            f"Question: {question}"
        )
        result = executor.invoke({"input": augmented_input, "chat_history": chat_history})

        steps = result.get("intermediate_steps", [])
        tools_used = [s[0].tool for s in steps]
        # Credit the forced pre-fetch even if the agent didn't explicitly
        # re-call video_content_search itself (expected — it was told not
        # to bother re-searching the same question) so downstream disclaimer
        # / eval logic that checks tools_used still sees it was used.
        if (
            not retrieved.startswith("No sufficiently relevant content found")
            and "video_content_search" not in tools_used
        ):
            steps = [(_FakeTool("video_content_search"), retrieved)] + steps

        return {"output": result["output"], "intermediate_steps": steps}

    return answer_question


class _FakeTool:
    """Minimal stand-in so answer_question can credit the forced
    pre-fetch in intermediate_steps the same shape AgentExecutor's real
    steps use (step[0].tool), without pulling in langchain's actual
    AgentAction type for a single-field shim."""
    def __init__(self, tool):
        self.tool = tool

    
    
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
    client, collection, video_index, chunks = init_clients()
    answer_question = build_agent(client, collection, video_index, chunks)

except Exception as e:
    st.error(f"Failed to initialize the assistant: {e}")
    st.stop()

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
                result = answer_question(question, st.session_state.chat_history)
                answer = result["output"]
                steps = result.get("intermediate_steps", [])
                tools_used = set(step[0].tool for step in steps)

                used_content = bool(st.session_state.last_sources)
                used_metadata = "video_library_metadata" in tools_used
                used_general = "general_procurement_knowledge" in tools_used or not steps

                parts = []
                if used_content:
                    parts.append("video content")
                if used_metadata:
                    parts.append("video metadata library")
                if used_general:
                    parts.append("general knowledge")

                if len(parts) > 1:
                    if len(parts) == 2:
                        joined = " and ".join(parts)
                    else:
                        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
                    label = f"(This answer combines {joined}.)"
                elif used_metadata:
                    label = "(Sourced from the video library's metadata — titles and topics, not transcript content.)"
                elif used_general:
                    label = "(General knowledge — not sourced from the video library.)"
                else:
                    label = None

                if label:
                    answer = f"{label}\n\n{answer}"
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
