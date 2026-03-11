"""API endpoint tests for the FastAPI application."""

import pytest


pytestmark = pytest.mark.api


class TestQueryEndpoint:

    def test_successful_query(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        mock_rag_system.query.assert_called_once()

    def test_query_with_session_id(self, client, mock_rag_system):
        resp = client.post(
            "/api/query",
            json={"query": "Follow up", "session_id": "existing_session"},
        )
        assert resp.status_code == 200
        args, kwargs = mock_rag_system.query.call_args
        assert args[1] == "existing_session"

    def test_query_creates_session_when_none_provided(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_missing_body(self, client):
        resp = client.post("/api/query")
        assert resp.status_code == 422

    def test_query_empty_string(self, client):
        resp = client.post("/api/query", json={"query": ""})
        # Empty string is a valid Pydantic str, endpoint should still process it
        assert resp.status_code == 200

    def test_query_returns_sources_with_url(self, client):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        data = resp.json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "MCP Course - Lesson 3"
        assert data["sources"][0]["url"] == "https://example.com/lesson"

    def test_query_internal_error(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")
        resp = client.post("/api/query", json={"query": "fail"})
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]


class TestCoursesEndpoint:

    def test_get_courses(self, client, mock_rag_system):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 2
        assert "MCP Course" in data["course_titles"]
        assert "AI Agents" in data["course_titles"]

    def test_courses_internal_error(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db down")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "db down" in resp.json()["detail"]


class TestSessionEndpoint:

    def test_clear_session(self, client, mock_rag_system):
        resp = client.delete("/api/sessions/session_1")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("session_1")


class TestStaticFiles:

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
