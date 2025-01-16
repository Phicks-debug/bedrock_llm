# flake8: noqa :E203
"""Base Agent implementation with shared functionality."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from .schema.message import MessageBlock
from .schema.tools import InputSchema, ToolMetadata
from .types.exceptions import ToolExecutionError


class BaseAgent(ABC):
    """Base agent class for shared functionality between sync/async implementations."""

    tool_functions: Dict[str, Dict[str, Any]] = {}
    _tool_cache: Dict[str, Any] = {}

    @classmethod
    def tool(cls, metadata: ToolMetadata):
        """Shared tool registration logic."""

        def decorator(func):
            cache_key = metadata.name
            if cache_key in cls._tool_cache:
                return cls._tool_cache[cache_key]

            try:
                metadata_dict = metadata.model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid tool metadata for {metadata.name}: {str(e)}")

            tool_info = {
                "function": func,
                "metadata": metadata_dict,
                "created_at": metadata_dict.get("created_at", None),
            }
            cls.tool_functions[metadata.name] = tool_info
            cls._tool_cache[cache_key] = func
            return func

        return decorator

    def __init__(
        self,
        auto_update_memory: bool = True,
        max_iterations: int = 5,
        memory_limit: Optional[int] = None,
    ):
        self.max_iterations = max_iterations
        self._memory_limit = memory_limit or 100
        self._conversation_history: List[MessageBlock] = []
        self.auto_update_memory = auto_update_memory

    def _validate_tool_params(
        self, tool_data: Dict[str, Any], params: Dict[str, Any]
    ) -> None:
        """Validate tool parameters against schema."""
        if "input_schema" in tool_data["metadata"]:
            try:
                InputSchema(**tool_data["metadata"]["input_schema"]).model_validate(
                    params
                )
            except ValidationError as e:
                raise ToolExecutionError(
                    tool_data["metadata"]["name"], f"Invalid parameters: {str(e)}"
                )

    def _manage_memory(self) -> None:
        """Manage conversation history size."""
        if len(self._conversation_history) > self._memory_limit:
            # Keep most recent messages
            self._conversation_history = self._conversation_history[
                -self._memory_limit :
            ]

    @lru_cache(maxsize=32)
    def _get_memory_update(self, prompt_str: str) -> Dict[str, Any]:
        """Cache memory updates for identical prompts."""
        return MessageBlock(role="user", content=prompt_str).model_dump()

    def _update_memory(
        self, prompt: Union[str, MessageBlock, Sequence[MessageBlock]]
    ) -> None:
        """Update conversation memory with new prompt."""
        if not isinstance(self.memory, list):
            raise ValueError("Memory must be a list")

        if isinstance(prompt, str):
            self.memory.append(self._get_memory_update(prompt))
        elif isinstance(prompt, MessageBlock):
            self.memory.append(prompt.model_dump())
        elif isinstance(prompt, (list, Sequence)):
            if all(isinstance(x, MessageBlock) for x in prompt):
                self.memory.extend(msg.model_dump() for msg in prompt)
            else:
                self.memory.extend(prompt)
        else:
            raise ValueError("Invalid prompt format")

        self._manage_memory()

    @abstractmethod
    def _format_tool_result(self, result: Any) -> str:
        """Format tool execution result for model consumption."""
        pass

    @abstractmethod
    def _handle_tool_error(self, error: Exception, tool_name: str) -> Tuple[str, bool]:
        """Handle tool execution errors."""
        pass
