# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
uv sync

# Run the app (starts FastAPI on port 8000)
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

Web UI: http://localhost:8000 | API docs: http://localhost:8000/docs

Requires `ANTHROPIC_API_KEY` in a `.env` file at the project root (see `.env.example`).

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using semantic search + Claude AI.

**Backend (`backend/`)** — Python FastAPI, all modules at the top level (no nested packages):

- `app.py` — FastAPI entry point. Two API routes: `POST /api/query` and `GET /api/courses`. Mounts `frontend/` as static files. On startup, loads all `.txt` files from `docs/` into the vector store.
- `rag_system.py` — Central orchestrator. Owns all components (document processor, vector store, AI generator, session manager, tool manager) and wires them together. The `query()` method is the main entry point for handling user questions.
- `ai_generator.py` — Claude API client. Sends the user query with tool definitions, handles the tool-use loop (up to one round: initial call → tool execution → follow-up call), returns final text.
- `search_tools.py` — Tool abstraction layer. `CourseSearchTool` wraps vector store search as an Anthropic tool definition. `ToolManager` registers tools, executes them by name, and tracks sources for the response.
- `vector_store.py` — ChromaDB wrapper with two collections: `course_catalog` (course metadata, used for semantic course name resolution) and `course_content` (chunked lesson text). The `search()` method handles course name resolution → filter building → semantic search in one call.
- `document_processor.py` — Parses structured text files (Course Title/Link/Instructor header, then `Lesson N: Title` markers) into `Course` + `CourseChunk` models. Chunks using sentence-aware splitting (800 chars, 100 overlap).
- `session_manager.py` — In-memory conversation history keyed by session ID. History is formatted as a string and appended to the system prompt (not as separate messages).
- `config.py` — Dataclass loading from `.env`. Model, chunk size, and other settings are here.

**Frontend (`frontend/`)** — Vanilla JS/HTML/CSS, no build step. Served as static files by FastAPI. Uses `marked.js` for markdown rendering.

**Course documents (`docs/`)** — Plain text files with a specific format (see `document_processor.py` for the parsing logic).

## Key Design Decisions

- **Tool-based search**: Claude decides whether to search via tool_choice=auto, rather than searching on every query. The tool definition is in `CourseSearchTool.get_tool_definition()`.
- **Conversation history is injected into the system prompt** as a formatted string, not as separate message turns in the API call.
- **Course name resolution**: When a user mentions a course by partial name, the vector store first queries `course_catalog` semantically to resolve the full title, then uses it as a metadata filter on `course_content`.
- **ChromaDB is persistent** (stored at `./chroma_db` relative to backend working directory). On startup, already-loaded courses are skipped by title deduplication.
