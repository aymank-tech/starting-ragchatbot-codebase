"""End-to-end tests for RAGSystem.query()."""

from unittest.mock import MagicMock, patch
from tests.helpers import make_text_response, make_tool_use_response
from rag_system import RAGSystem


def _make_rag(mock_client):
    """Build a RAGSystem with mocked VectorStore and Anthropic client."""
    mock_config = MagicMock()
    mock_config.CHUNK_SIZE = 800
    mock_config.CHUNK_OVERLAP = 100
    mock_config.CHROMA_PATH = "./test_chroma"
    mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    mock_config.MAX_RESULTS = 5
    mock_config.ANTHROPIC_API_KEY = "fake-key"
    mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    mock_config.MAX_HISTORY = 2

    with (
        patch("rag_system.VectorStore"),
        patch("rag_system.DocumentProcessor"),
        patch(
            (
                "rag_system.anthropic.Anthropic"
                if False
                else "ai_generator.anthropic.Anthropic"
            ),
            return_value=mock_client,
        ),
    ):
        rag = RAGSystem(mock_config)

    return rag


class TestRAGSystemQuery:

    def test_returns_tuple_of_str_and_list(self):
        client = MagicMock()
        client.messages.create.return_value = make_text_response("Answer")

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore"),
            patch("rag_system.DocumentProcessor"),
        ):
            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )
            response, sources = rag.query("What is MCP?")

        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_sources_are_list_of_dicts_not_strings(self):
        """Bug #1: Return annotation says List[str] but actual sources are List[Dict]."""
        client = MagicMock()
        client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            make_text_response("MCP info here."),
        ]

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore") as MockVS,
            patch("rag_system.DocumentProcessor"),
        ):
            # Configure the mock vector store instance
            vs_instance = MockVS.return_value
            vs_instance.search.return_value = MagicMock(
                error=None,
                is_empty=lambda: False,
                documents=["MCP overview"],
                metadata=[
                    {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0}
                ],
            )
            vs_instance.get_lesson_link.return_value = "https://example.com/l1"
            vs_instance.get_course_link.return_value = "https://example.com/c"

            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )
            _, sources = rag.query("What is MCP?")

        # Sources should be dicts with text/url keys (not plain strings)
        assert len(sources) > 0
        assert isinstance(sources[0], dict)
        assert "text" in sources[0]
        assert "url" in sources[0]

    def test_query_wrapped_in_prompt_template(self):
        client = MagicMock()
        client.messages.create.return_value = make_text_response("Answer")

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore"),
            patch("rag_system.DocumentProcessor"),
        ):
            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )
            rag.query("What is MCP?")

        call_kwargs = client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        assert "Answer this question about course materials:" in user_msg
        assert "What is MCP?" in user_msg

    def test_session_history_passed_to_generator(self):
        client = MagicMock()
        client.messages.create.return_value = make_text_response("Answer")

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore"),
            patch("rag_system.DocumentProcessor"),
        ):
            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )
            # Create a session with history
            sid = rag.session_manager.create_session()
            rag.session_manager.add_exchange(sid, "Hi", "Hello!")
            rag.query("Follow up", session_id=sid)

        call_kwargs = client.messages.create.call_args.kwargs
        assert "Previous conversation:" in call_kwargs["system"]
        assert "Hi" in call_kwargs["system"]

    def test_sources_reset_after_retrieval(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            make_text_response("Answer"),
        ]

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore") as MockVS,
            patch("rag_system.DocumentProcessor"),
        ):
            vs_instance = MockVS.return_value
            vs_instance.search.return_value = MagicMock(
                error=None,
                is_empty=lambda: False,
                documents=["text"],
                metadata=[{"course_title": "C", "lesson_number": 1, "chunk_index": 0}],
            )
            vs_instance.get_lesson_link.return_value = None
            vs_instance.get_course_link.return_value = "https://example.com"

            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )
            rag.query("Q")

        # After query, sources should have been reset
        assert rag.tool_manager.get_last_sources() == []

    def test_no_tool_use_returns_empty_sources(self):
        client = MagicMock()
        client.messages.create.return_value = make_text_response("General answer")

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore"),
            patch("rag_system.DocumentProcessor"),
        ):
            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )
            _, sources = rag.query("What is Python?")

        assert sources == []

    def test_both_tools_registered(self):
        client = MagicMock()
        client.messages.create.return_value = make_text_response("ok")

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=client),
            patch("rag_system.VectorStore"),
            patch("rag_system.DocumentProcessor"),
        ):
            rag = RAGSystem(
                MagicMock(
                    CHUNK_SIZE=800,
                    CHUNK_OVERLAP=100,
                    CHROMA_PATH="./t",
                    EMBEDDING_MODEL="m",
                    MAX_RESULTS=5,
                    ANTHROPIC_API_KEY="k",
                    ANTHROPIC_MODEL="m",
                    MAX_HISTORY=2,
                )
            )

        tool_names = {d["name"] for d in rag.tool_manager.get_tool_definitions()}
        assert tool_names == {"search_course_content", "get_course_outline"}
