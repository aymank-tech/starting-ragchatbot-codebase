import pytest
from unittest.mock import MagicMock
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
    mock_client.messages.create.return_value = make_text_response(
        "Default mock answer."
    )
    monkeypatch.setattr(
        "ai_generator.anthropic.Anthropic", lambda **kwargs: mock_client
    )
    return mock_client
