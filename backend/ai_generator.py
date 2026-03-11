import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about a course's structure, outline, syllabus, lesson list, or what topics a course covers. When returning outline information, include the course title, course link, and the full lesson list with each lesson's number and title.
- **Up to 2 sequential tool calls per query** — use a second call only when the first result is insufficient or when you need information from a different tool to fully answer the question
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use the appropriate tool first, then answer
- **Outline/structure questions**: Use get_course_outline and present the course title, link, and complete lesson listing
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls per query.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters — reused for every call in the loop
        messages = [{"role": "user", "content": query}]
        api_params = {**self.base_params, "system": system_content}
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Initial API call
        response = self.client.messages.create(**api_params, messages=messages)

        # Tool-use loop: execute tools and call API again, up to MAX_TOOL_ROUNDS
        rounds_used = 0
        while (
            response.stop_reason == "tool_use"
            and tool_manager
            and rounds_used < self.MAX_TOOL_ROUNDS
        ):
            rounds_used += 1

            # Append Claude's tool-use response
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool_use block and collect results
            tool_results = []
            tool_error = False
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                    except Exception as exc:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Tool execution failed: {exc}",
                                "is_error": True,
                            }
                        )
                        tool_error = True

            messages.append({"role": "user", "content": tool_results})

            # Call API again with accumulated messages
            response = self.client.messages.create(**api_params, messages=messages)

            # Stop looping after a tool error (Claude already received the error)
            if tool_error:
                break

        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """Extract the first text block from a response, or return a fallback."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return "I wasn't able to generate a response."
