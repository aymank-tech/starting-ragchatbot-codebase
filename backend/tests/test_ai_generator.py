"""Tests for AIGenerator tool-calling integration."""

from unittest.mock import MagicMock, call
from ai_generator import AIGenerator
from tests.helpers import make_text_response, make_tool_use_response


TOOLS = [{"name": "search_course_content"}, {"name": "get_course_outline"}]


class TestAIGeneratorDirectResponse:
    """Tests where Claude returns text without using any tools."""

    def test_returns_text_content(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_text_response("Hello!")
        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")

        result = gen.generate_response(query="Hi")
        assert result == "Hello!"

    def test_single_api_call_when_no_tools(self, mock_anthropic_client):
        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="Hi")

        assert mock_anthropic_client.messages.create.call_count == 1

    def test_conversation_history_appended_to_system(self, mock_anthropic_client):
        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="Hi", conversation_history="User: Hello\nAssistant: Hey")

        api_call = mock_anthropic_client.messages.create.call_args
        system = api_call.kwargs["system"]
        assert "Previous conversation:" in system
        assert "User: Hello" in system

    def test_no_history_means_base_prompt_only(self, mock_anthropic_client):
        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="Hi", conversation_history=None)

        api_call = mock_anthropic_client.messages.create.call_args
        system = api_call.kwargs["system"]
        assert "Previous conversation:" not in system

    def test_tools_passed_with_auto_choice(self, mock_anthropic_client):
        tools = [{"name": "search", "description": "search", "input_schema": {}}]
        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="search for X", tools=tools)

        api_call = mock_anthropic_client.messages.create.call_args
        assert api_call.kwargs["tools"] == tools
        assert api_call.kwargs["tool_choice"] == {"type": "auto"}


class TestAIGeneratorSingleToolRound:
    """Tests for the single tool-use → execute → follow-up path."""

    def test_single_round_calls_tool_and_returns_followup(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            make_text_response("Here is what I found about MCP."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[MCP Course - Lesson 1]\nMCP overview..."

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="Tell me about MCP",
            tools=TOOLS,
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="MCP")
        assert mock_anthropic_client.messages.create.call_count == 2
        assert result == "Here is what I found about MCP."

    def test_tools_included_in_followup_call(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "X"}),
            make_text_response("Answer"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "results"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="Q", tools=TOOLS, tool_manager=tool_manager)

        # Both initial and follow-up calls should include tools
        for c in mock_anthropic_client.messages.create.call_args_list:
            assert c.kwargs["tools"] == TOOLS
            assert c.kwargs["tool_choice"] == {"type": "auto"}

    def test_tool_use_without_tool_manager_returns_fallback(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_tool_use_response(
            "search_course_content", {"query": "test"},
        )
        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")

        result = gen.generate_response(
            query="search", tools=TOOLS, tool_manager=None,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestAIGeneratorMultiToolRounds:
    """Tests for sequential multi-round tool calling (up to 2 rounds)."""

    def test_two_sequential_tool_rounds(self, mock_anthropic_client):
        """Claude calls get_course_outline, then search_course_content, then answers."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("get_course_outline", {"course_name": "MCP"}, tool_id="toolu_1"),
            make_tool_use_response("search_course_content", {"query": "tools"}, tool_id="toolu_2"),
            make_text_response("MCP covers tools in lesson 3."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            "Course: MCP\nLesson 3: Tools",
            "[MCP Course - Lesson 3]\nTool usage details...",
        ]

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(
            query="What does MCP say about tools?",
            tools=TOOLS,
            tool_manager=tool_manager,
        )

        assert mock_anthropic_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        assert tool_manager.execute_tool.call_args_list[0] == call("get_course_outline", course_name="MCP")
        assert tool_manager.execute_tool.call_args_list[1] == call("search_course_content", query="tools")
        assert result == "MCP covers tools in lesson 3."

    def test_stops_after_max_rounds(self, mock_anthropic_client):
        """Even if Claude keeps requesting tools, only 2 rounds are executed."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "a"}, tool_id="toolu_1"),
            make_tool_use_response("search_course_content", {"query": "b"}, tool_id="toolu_2"),
            # Third response is also tool_use — but rounds are exhausted
            make_tool_use_response("search_course_content", {"query": "c"}, tool_id="toolu_3"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some result"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(query="Q", tools=TOOLS, tool_manager=tool_manager)

        # 3 API calls: initial + 2 follow-ups
        assert mock_anthropic_client.messages.create.call_count == 3
        # Only 2 tool executions (third tool_use response is NOT executed)
        assert tool_manager.execute_tool.call_count == 2
        # Fallback because final response has no TextBlock
        assert result == "I wasn't able to generate a response."

    def test_messages_accumulate_across_rounds(self, mock_anthropic_client):
        """Verify conversation context builds up correctly over 2 rounds.

        messages is a mutable list shared across calls, so we capture
        snapshots via a side_effect to inspect each call's state.
        """
        responses = [
            make_tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="toolu_1"),
            make_tool_use_response("search_course_content", {"query": "Y"}, tool_id="toolu_2"),
            make_text_response("Final answer"),
        ]
        snapshots = []

        def capture_and_respond(**kwargs):
            # Snapshot the messages list at each call
            snapshots.append(list(kwargs["messages"]))
            return responses.pop(0)

        mock_anthropic_client.messages.create.side_effect = capture_and_respond
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline data", "search data"]

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="Q", tools=TOOLS, tool_manager=tool_manager)

        # Call 0 (initial): just the user message
        assert len(snapshots[0]) == 1
        assert snapshots[0][0]["role"] == "user"

        # Call 1 (after round 1): user + assistant(tool_use) + user(tool_result)
        assert len(snapshots[1]) == 3
        assert snapshots[1][1]["role"] == "assistant"
        assert snapshots[1][2]["role"] == "user"
        assert snapshots[1][2]["content"][0]["type"] == "tool_result"
        assert snapshots[1][2]["content"][0]["tool_use_id"] == "toolu_1"

        # Call 2 (after round 2): 5 messages total
        assert len(snapshots[2]) == 5
        assert snapshots[2][3]["role"] == "assistant"
        assert snapshots[2][4]["role"] == "user"
        assert snapshots[2][4]["content"][0]["tool_use_id"] == "toolu_2"

    def test_tools_included_in_all_api_calls(self, mock_anthropic_client):
        """Tools and tool_choice must be present in every API call, not stripped."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "a"}, tool_id="toolu_1"),
            make_tool_use_response("search_course_content", {"query": "b"}, tool_id="toolu_2"),
            make_text_response("Done"),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "data"

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        gen.generate_response(query="Q", tools=TOOLS, tool_manager=tool_manager)

        for c in mock_anthropic_client.messages.create.call_args_list:
            assert c.kwargs["tools"] == TOOLS
            assert c.kwargs["tool_choice"] == {"type": "auto"}


class TestAIGeneratorToolErrors:
    """Tests for graceful handling of tool execution errors."""

    def test_tool_exception_sent_as_error_result(self, mock_anthropic_client):
        """When execute_tool raises, the error is sent to Claude and the loop stops."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "bad"}, tool_id="toolu_err"),
            make_text_response("Sorry, the search failed."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("ChromaDB connection lost")

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(query="Q", tools=TOOLS, tool_manager=tool_manager)

        assert mock_anthropic_client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1
        assert result == "Sorry, the search failed."

        # Verify the error tool_result was sent to Claude
        second_call_msgs = mock_anthropic_client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_msg = second_call_msgs[-1]["content"][0]
        assert tool_result_msg["is_error"] is True
        assert "ChromaDB connection lost" in tool_result_msg["content"]

    def test_error_on_second_round_stops_loop(self, mock_anthropic_client):
        """First round succeeds, second round tool raises — loop stops, Claude responds."""
        mock_anthropic_client.messages.create.side_effect = [
            make_tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="toolu_1"),
            make_tool_use_response("search_course_content", {"query": "Y"}, tool_id="toolu_2"),
            make_text_response("Partial results: outline worked but search failed."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            "Course: X\nLessons...",
            RuntimeError("DB error"),
        ]

        gen = AIGenerator(api_key="fake", model="claude-sonnet-4-20250514")
        result = gen.generate_response(query="Q", tools=TOOLS, tool_manager=tool_manager)

        assert mock_anthropic_client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        assert result == "Partial results: outline worked but search failed."
