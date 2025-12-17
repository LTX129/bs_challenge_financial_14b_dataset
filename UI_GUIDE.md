# Financial RAG Web UI

This adds a small FastAPI + Vue.js UI so you can run the RAG workflow in a browser (question picker + free-form questions, backend toggle between local and remote LLMs).

## Quick start

1. Install FastAPI + Uvicorn (and your LangChain deps):  
   `pip install fastapi uvicorn`
2. Make sure your Chroma store is ready (run the existing CLI import helpers if needed).
3. Start the API + UI (use 7860 to avoid clashing with a Chroma HTTP server on 8000):  
   `uvicorn api.server:app --host 0.0.0.0 --port 7860`
4. Open `http://localhost:7860/` to use the UI.

## What you get

- **/api/ask** – POST `{question, backend?, k?, mutuality?, question_id?}` returns `answer`, `sources`, `hits`, and the backend used.  
- **/api/questions** – GET with `q`, `limit`, `offset` to browse/search `question.json`.  
- **/api/status** – GET basic config (default backend, total questions, collection name).  
- Static Vue UI served from `/` (and `/ui`) with:
  - Backend switcher (`local` vs `qwen` via util.py).
  - Preset question picker (search/pagination over `question.json`).
  - Free-form ask box with top-k and similarity threshold sliders.
  - Answer panel with source snippets.

## Backend selection (local vs remote)

- Default comes from `RAG_MODEL_BACKEND` (`local` or `qwen`).  
- The UI request body can override with `backend`.  
- **Important:** Embeddings default to local to match existing Chroma collections. If you need remote embeddings, set `RAG_EMBED_BACKEND=remote` (or `qwen`), but ensure the collection is built with the same embedding dimension.  
- For local models, set `LOCAL_LLM_BACKEND` (`ollama` or `openai`-compatible), `LOCAL_LLM_MODEL`, `LOCAL_LLM_BASE_URL`, and embedding envs in `conf/.env`.  
- For remote Qwen, keep your DashScope credentials in `conf/.qwen` or env vars. ChatTongyi is used for prompts (llm/chat are the same).

## Other knobs

- `QUESTION_PATH` – override the preset question file (defaults to `question.json`).  
- `CHROMA_SERVER_TYPE` – `local` (default) or `http` to point at a remote Chroma server. Pair with `CHROMA_COLLECTION` and `CHROMA_PERSIST_PATH` envs.  
- `k` and `mutuality` (0–1) are exposed in the UI and forwarded to the retriever.

## Notes

- The Vue UI uses the CDN build of Vue 3; keep internet access available for that or swap in a local bundle.  
- Logging and retrieval behavior remain the same as the CLI flow; the API only wraps `RagManager.get_result_with_sources`.  
- If you change ports or hosts, update your browser URL; the UI talks to the same origin (`/api/...`) by default.
