"""Shared test helpers for building mock Anthropic responses."""

import anthropic.types as t


def _make_message(
    content_blocks,
    stop_reason="end_turn",
    role="assistant",
    model="claude-sonnet-4-20250514",
):
    """Build a real anthropic.types.Message from content blocks."""
    return t.Message(
        id="msg_test_123",
        type="message",
        role=role,
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=t.Usage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def make_text_response(text: str):
    """Return an anthropic Message whose first content block is TextBlock."""
    return _make_message([t.TextBlock(type="text", text=text)])


def make_tool_use_response(
    tool_name: str, tool_input: dict, tool_id: str = "toolu_test_1"
):
    """Return an anthropic Message whose first content block is ToolUseBlock."""
    return _make_message(
        [t.ToolUseBlock(type="tool_use", id=tool_id, name=tool_name, input=tool_input)],
        stop_reason="tool_use",
    )
