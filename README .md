# 📦 Procurement-SCM- Intelligence Assistant

## About

A RAG-based chatbot that answers procurement, sourcing, and supply chain management questions grounded in a curated library of expert video content — with a general-knowledge fallback for questions the videos don't cover. Built as a LangChain tool-calling agent (not a single-shot RAG chain), so it can route between video-content retrieval, library metadata, and general domain knowledge per question, and combine more than one when a question needs it.

**Sample interaction:**
> **Q:** How many videos discuss sustainability, and briefly explain what sustainable procurement means?
> **A:** *(combines video_library_metadata + general_procurement_knowledge)* — "In the video library, there is one video that discusses sustainability... Sustainable procurement refers to the process of acquiring goods and services while considering not only economic factors but also environmental and social impacts..."

**Live app:** deployed on Streamlit Community Cloud (see repo for current URL).

## Problem statement

Procurement and SCM learners often have access to good video content but no easy way to query it — they either rewatch hours of footage or fall back on generic chatbot answers that aren't grounded in the material. This project builds an assistant that:
- Answers specific questions using only what's actually said in the indexed videos, with source citations and timestamped links
- Falls back cleanly to general domain knowledge when a question is outside the video library, and labels that answer as such
- Refuses out-of-scope, harmful, or prompt-injected requests
- Supports both text and voice input

## Dataset

- **Source:** 10 curated YouTube videos covering procurement, sourcing, contracts, inventory, demand planning, logistics, risk management, supplier relationship management, sustainability, and AI in procurement
- **Processing:** Whisper transcription → merged into ~250-token chunks with ~40-token overlap (so ideas spanning a boundary, e.g. a numbered list, still appear intact in at least one chunk)
- **Size:** 160 chunks total across 10 videos (~16 chunks/video average)
- **Metadata per chunk:** `video_id`, `title`, `start`/`end` timestamp — used for citations and clickable timestamped YouTube links
- **License/attribution:** transcripts derived from publicly available YouTube content; two videos were pre-trimmed to remove non-substantive webinar preamble before indexing

## Architecture

```
User question
     │
     ▼
Forced pre-fetch: video_content_search runs first, always
     │  (embeddings + BM25 hybrid retrieval → Cohere rerank → top-8 chunks)
     ▼
Tool-calling agent (gpt-4o-mini, temperature=0)
     │
     ├─ video_content_search        (curated video transcripts, hybrid retrieval)
     ├─ video_library_metadata      (video count / topic coverage, no semantic search)
     └─ general_procurement_knowledge (LLM's own domain knowledge, clearly labeled)
     │
     ▼
Answer + source citations (if video-grounded) + provenance label
```

**Retrieval pipeline:** OpenAI `text-embedding-3-small` embeddings + BM25 keyword search merged into a candidate pool, reranked with Cohere's `rerank-english-v3.0`, filtered by relevance threshold, capped at top-8. Hybrid search was added specifically to catch exact-phrase/proper-noun matches (e.g. tool names like "Icertis") that pure embedding search missed.

**Why retrieval is forced, not agent-discretionary:** early testing showed the agent would sometimes skip calling `video_content_search` entirely on questions that sound like well-known business frameworks, answering from training data instead — even with an explicit "search first" instruction. Retrieval now always runs in code before the agent's turn, removing that failure mode at the source rather than relying on prompting alone.

**Voice input:** OpenAI Whisper (`whisper-1`) transcribes uploaded audio, which is then handled as a normal text question.

## Results

Evaluated with LangSmith against a 25-case fixed test set spanning video-content, general-knowledge, metadata, compound (multi-tool), and adversarial (out-of-scope / harmful / prompt-injection) questions, using four evaluators (routing, source, refusal, and answer correctness — the latter two LLM-judged).

| Metric | Result |
|---|---|
| Routing correctness | 100% |
| Source correctness | 100% |
| Refusal correctness | 100% |
| Answer correctness | ~83% |
| Latency (P50) | ~14s |

Six structural bugs were found and fixed during testing (forced retrieval, compound-question tool coverage, unretrieved-fact padding, a count-presupposition prompt bug, weak-match retrieval contamination, and app/eval-script config drift). Remaining correctness gaps are two cases confirmed via repeated (5x) runs to be genuine LLM output non-determinism on long transcript segments, not defects — see `evaluation_report.md` for full methodology and per-case analysis.

## Setup & installation

```bash
git clone <repo-url>
cd Project-4-Procurement-SCM-Intelligent-Assistant

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here        # optional — enables hybrid rerank; degrades gracefully without it
LANGCHAIN_API_KEY=your_key_here     # optional — enables LangSmith tracing/eval
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=procurement-scm-assistant
```

Run the app:
```bash
streamlit run app.py
```
On first run, it builds the Chroma vector index from `data/chunks.json` (index isn't committed to the repo — it rebuilds automatically).

Run the evaluation suite:
```bash
python langsmith_eval.py
```

## Project structure

```
app.py                          # Streamlit app — deployed entry point
langsmith_eval.py               # LangSmith evaluation harness
Procurement_SCM_assistant.ipynb # MVP notebook — chunking, embedding, agent build, sanity checks
eval_set.json                   # 25-case fixed evaluation dataset
data/chunks.json                # processed transcript chunks (source for both app and eval)
requirements.txt
```

## Tech stack

Python · LangChain (tool-calling agent) · OpenAI (`gpt-4o-mini`, `text-embedding-3-small`, `whisper-1`) · ChromaDB (vector store) · Cohere (`rerank-english-v3.0`) · `rank-bm25` (keyword search) · Streamlit (frontend) · LangSmith (evaluation & tracing)

## Future improvements

- Consolidate the duplicated agent-building logic across `app.py`, `langsmith_eval.py`, and the notebook into a single shared module
- Expand the video library beyond 10 videos for broader topic coverage
- Investigate the two remaining flaky-correctness cases further (long-segment summarization variance)

## Author

David Tayebwa — MSc Procurement, Logistics & Supply Chain Management (University of Salford), MCIPS · Ironhack AI Engineering Bootcamp (cohort AI26)
