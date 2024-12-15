from typing import Optional


class AsyncClientSessionError(TypeError):
    pass


class ModelMemoryException(TypeError):
    pass


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""

    def __init__(
        self, tool_name: str, message: str, original_error: Optional[Exception] = None
    ) -> None:
        self.tool_name = tool_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"Error in tool '{tool_name}': {message}")
