"""Tests for CourseSearchTool, CourseOutlineTool, and ToolManager."""

from unittest.mock import MagicMock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults

# ===================================================================
# CourseSearchTool.execute()
# ===================================================================


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() result formatting & filtering."""

    def test_formatted_output_contains_headers_and_text(
        self, mock_vector_store, sample_search_results
    ):
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="tools")

        assert "[MCP Course - Lesson 3]" in result
        assert "[AI Agents - Lesson 1]" in result
        assert "MCP allows AI models" in result
        assert "Agents use a loop" in result

    def test_passes_course_name_filter(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="tools",
            course_name="MCP",
            lesson_number=None,
        )

    def test_passes_lesson_number_filter(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="tools",
            course_name=None,
            lesson_number=3,
        )

    def test_passes_both_filters(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools", course_name="MCP", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="tools",
            course_name="MCP",
            lesson_number=3,
        )

    # -- Empty results messages --

    def test_empty_results_no_filters(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="quantum computing")

        assert result == "No relevant content found."

    def test_empty_results_with_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="quantum", course_name="Physics")

        assert "in course 'Physics'" in result

    def test_empty_results_with_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="quantum", lesson_number=5)

        assert "in lesson 5" in result

    def test_empty_results_with_both_filters(
        self, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="quantum", course_name="Physics", lesson_number=5)

        assert "in course 'Physics'" in result
        assert "in lesson 5" in result

    # -- Error results --

    def test_error_results_return_error_string(
        self, mock_vector_store, error_search_results
    ):
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything")

        assert result == "No course found matching 'nonexistent'"

    # -- Source tracking --

    def test_last_sources_text_and_url(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "MCP Course - Lesson 3"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson"

    def test_source_falls_back_to_course_link(self, mock_vector_store):
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="tools")

        # Should fall back to course link when lesson link is None
        assert tool.last_sources[0]["url"] == "https://example.com/course"

    def test_source_without_lesson_number(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=["Some content"],
            metadata=[
                {"course_title": "Intro", "lesson_number": None, "chunk_index": 0}
            ],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="intro")

        assert "[Intro]" in result  # no lesson suffix in header
        assert tool.last_sources[0]["text"] == "Intro"
        # lesson link should NOT be requested when lesson_number is None
        mock_vector_store.get_lesson_link.assert_not_called()


class TestCourseSearchToolDefinition:

    def test_tool_definition_schema(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        defn = tool.get_tool_definition()

        assert defn["name"] == "search_course_content"
        props = defn["input_schema"]["properties"]
        assert "query" in props
        assert "course_name" in props
        assert "lesson_number" in props
        assert defn["input_schema"]["required"] == ["query"]


# ===================================================================
# CourseOutlineTool
# ===================================================================


class TestCourseOutlineTool:

    def test_formatted_outline(self, mock_vector_store):
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="MCP")

        assert "Course: MCP Course" in result
        assert "Link: https://example.com/mcp" in result
        assert "Lessons (3 total):" in result
        assert "Lesson 1: Introduction" in result
        assert "Lesson 2: Architecture" in result
        assert "Lesson 3: Tools" in result

    def test_not_found(self, mock_vector_store):
        mock_vector_store.get_course_outline.return_value = None
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Nonexistent")

        assert "No course found matching 'Nonexistent'" in result

    def test_outline_without_link(self, mock_vector_store):
        mock_vector_store.get_course_outline.return_value = {
            "title": "Bare Course",
            "course_link": None,
            "lessons": [{"lesson_number": 1, "lesson_title": "Only Lesson"}],
        }
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Bare")

        assert "Course: Bare Course" in result
        assert "Link:" not in result  # no link line when link is None


# ===================================================================
# ToolManager
# ===================================================================


class TestToolManager:

    def test_register_and_get_definitions(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))

        defs = tm.get_tool_definitions()
        names = {d["name"] for d in defs}
        assert names == {"search_course_content", "get_course_outline"}

    def test_execute_by_name(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        result = tm.execute_tool("search_course_content", query="tools")

        assert "[MCP Course" in result

    def test_execute_unknown_tool(self):
        tm = ToolManager()
        result = tm.execute_tool("nonexistent_tool")

        assert "not found" in result

    def test_get_last_sources(self, mock_vector_store):
        tm = ToolManager()
        search = CourseSearchTool(mock_vector_store)
        tm.register_tool(search)
        tm.execute_tool("search_course_content", query="tools")

        sources = tm.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["text"] == "MCP Course - Lesson 3"

    def test_reset_sources(self, mock_vector_store):
        tm = ToolManager()
        search = CourseSearchTool(mock_vector_store)
        tm.register_tool(search)
        tm.execute_tool("search_course_content", query="tools")

        tm.reset_sources()
        assert tm.get_last_sources() == []

    def test_get_last_sources_empty_when_no_search(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        # No execute call
        assert tm.get_last_sources() == []
