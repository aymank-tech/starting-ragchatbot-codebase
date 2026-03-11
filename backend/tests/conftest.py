import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from vector_store import SearchResults
from tests.helpers import make_text_response, make_tool_use_response  # noqa: F401


# ---------------------------------------------------------------------------
# Search result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_search_results():
    """Search results with two hits from different courses/lessons."""
    return SearchResults(
        documents=[
            "MCP allows AI models to interact with external tools.",
            "Agents use a loop of observe-think-act to solve tasks.",
        ],
        metadata=[
            {"course_title": "MCP Course", "lesson_number": 3, "chunk_index": 0},
            {"course_title": "AI Agents", "lesson_number": 1, "chunk_index": 2},
        ],
        distances=[0.25, 0.40],
    )


@pytest.fixture
def empty_search_results():
    """Search results with no hits and no error."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Search results representing an error condition."""
    return SearchResults.empty("No course found matching 'nonexistent'")


# ---------------------------------------------------------------------------
# Mock VectorStore
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(sample_search_results):
    """A MagicMock standing in for VectorStore with sensible defaults."""
    store = MagicMock()
    store.search.return_value = sample_search_results
    store.get_course_link.return_value = "https://example.com/course"
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store.get_course_outline.return_value = {
        "title": "MCP Course",
        "course_link": "https://example.com/mcp",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction"},
            {"lesson_number": 2, "lesson_title": "Architecture"},
            {"lesson_number": 3, "lesson_title": "Tools"},
        ],
    }
    return store


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Patch anthropic.Anthropic so no real API calls are made.

    Returns the mock client so tests can configure `messages.create`.
    """
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_text_response("Default mock answer.")
    monkeypatch.setattr("ai_generator.anthropic.Anthropic", lambda **kwargs: mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# Mock RAG system fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem with sensible defaults."""
    rag = MagicMock()
    rag.query.return_value = (
        "MCP allows AI models to interact with external tools.",
        [{"text": "MCP Course - Lesson 3", "url": "https://example.com/lesson"}],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["MCP Course", "AI Agents"],
    }
    rag.session_manager = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    return rag


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def client(mock_rag_system):
    """FastAPI TestClient with the app-level RAGSystem replaced by a mock.

    Patches ``app.rag_system`` so no real vector store, embedding model,
    or Anthropic API calls are made during API tests.  The working directory
    is temporarily changed to ``backend/`` so relative paths in app.py
    (``../frontend``, ``../docs``) resolve correctly.
    """
    backend_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    prev_cwd = os.getcwd()
    os.chdir(backend_dir)

    # Remove cached app module so it re-imports with our patches active
    sys.modules.pop("app", None)
    try:
        with patch("rag_system.VectorStore"), \
             patch("rag_system.DocumentProcessor"), \
             patch("ai_generator.anthropic.Anthropic"):
            from app import app
            # Replace the module-level rag_system with our mock
            import app as app_module
            app_module.rag_system = mock_rag_system

            with TestClient(app, raise_server_exceptions=False) as tc:
                yield tc
    finally:
        sys.modules.pop("app", None)
        os.chdir(prev_cwd)
