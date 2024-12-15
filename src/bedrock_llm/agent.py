# flake8: noqa: E203
import json
from functools import lru_cache, wraps
from typing import (Any, Callable, Dict, Generator, List, Optional, Sequence,
                    Tuple, Union, cast)

from pydantic import ValidationError

from .client import Client
from .config.base import RetryConfig
from .config.model import ModelConfig
from .schema.message import (MessageBlock, TextBlock, ToolCallBlock,
                             ToolResultBlock, ToolUseBlock)
from .schema.response import ResponseBlock
from .schema.tools import InputSchema, ToolMetadata
from .types.enums import ModelName, StopReason, ToolState
from .types.exceptions import ToolExecutionError


class Agent(Client):

    tool_functions: Dict[str, Dict[str, Any]] = {}
    _tool_cache: Dict[str, Any] = {}

    @classmethod
    def tool(cls, metadata: ToolMetadata):
        def decorator(func):
            cache_key = metadata.name
            if cache_key in cls._tool_cache:
                return cls._tool_cache[cache_key]

            # Validate tool metadata
            try:
                metadata_dict = metadata.model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid tool metadata for {metadata.name}: {str(e)}")

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            tool_info = {
                "function": wrapper,
                "metadata": metadata_dict,
            }
            cls.tool_functions[metadata.name] = tool_info
            cls._tool_cache[cache_key] = wrapper

            # cls._logger.info(f"Registered tool: {metadata.name}")
            return wrapper

        return decorator

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        auto_update_memory: bool = True,
        max_iterations: Optional[int] = 5,
        retry_config: Optional[RetryConfig] = None,
        memory_limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(region_name, model_name, [], retry_config, **kwargs)
        self.max_iterations = max_iterations
        self._memory_limit = memory_limit or 100
        self._conversation_history: List[MessageBlock] = []
        self.auto_update_memory = auto_update_memory

    def __execute_tool(
        self, tool_data: Dict[str, Any], params: Dict[str, Any]
    ) -> Tuple[Any, bool]:
        """Execute a single tool synchronously."""
        tool_name = tool_data.get("metadata", {}).get("name", "unknown_tool")
        try:
            # Validate input parameters against schema
            if "input_schema" in tool_data["metadata"]:
                try:
                    InputSchema(**tool_data["metadata"]["input_schema"]).model_validate(
                        params
                    )
                except ValidationError as e:
                    raise ToolExecutionError(tool_name, f"Invalid parameters: {str(e)}")

            result = tool_data["function"](**params)

            # Handle different return types
            if isinstance(result, (dict, list)):
                return json.dumps(result), False
            return str(result), False

        except ToolExecutionError as e:
            raise ToolExecutionError(tool_name, str(e))
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            raise ToolExecutionError(tool_name, error_msg, e)

    def __process_tools(
        self, tools_list: Union[List[ToolUseBlock], List[ToolCallBlock]]
    ) -> Union[MessageBlock, List[MessageBlock]]:
        """Process tool use requests synchronously."""
        tool_state = (
            ToolState.CLAUDE
            if isinstance(tools_list[-1], ToolUseBlock)
            else ToolState.MISTRAL_JAMBA_LLAMA
        )

        if tool_state == ToolState.CLAUDE:
            message = MessageBlock(role="user", content=[])
        else:
            message: List[MessageBlock] = []

        # Process tools sequentially
        for tool in tools_list:
            if not isinstance(tool, (ToolUseBlock, ToolCallBlock)):
                continue

            if tool_state == ToolState.CLAUDE:
                tool_name = tool.name
                tool_data = self.tool_functions.get(tool_name)
                if tool_data:
                    result, is_error = self.__execute_tool(tool_data, tool.input)
                    if isinstance(message.content, list):
                        message.content.append(
                            ToolResultBlock(
                                type="tool_result",
                                tool_use_id=tool.id,
                                is_error=is_error,
                                content=[TextBlock(type="text", text=str(result))],
                            )
                        )
            else:
                tool_name = tool.function
                tool_params = json.loads(tool_name["arguments"])
                tool_data = self.tool_functions.get(tool_name["name"])
                if tool_data:
                    result, is_error = self.__execute_tool(tool_data, tool_params)
                    message.append(
                        MessageBlock(
                            role="tool",
                            name=tool.function["name"],
                            content=str(result),
                            tool_call_id=tool.id,
                        )
                    )

        return message

    def generate_and_action(
        self,
        prompt: Union[str, MessageBlock, Sequence[MessageBlock]],
        tools: Union[List[Callable], Any],
        system: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> Generator[
        Tuple[
            Optional[ResponseBlock],
            Optional[Union[ToolResultBlock, List[ToolResultBlock]]],
        ],
        None,
        None,
    ]:
        """Generate responses and perform actions based on prompt and tools."""
        if not isinstance(self.memory, list):
            raise ValueError("Memory must be a list")

        if self.auto_update_memory:
            self._update_memory(prompt)

        tool_metadata = None

        if tools:
            tool_names = [func.__name__ for func in tools]
            tool_metadata = [
                self.tool_functions[name]["metadata"]
                for name in tool_names
                if name in self.tool_functions
            ]

        if self.max_iterations is None:
            raise ValueError("max_iterations must not be None")

        for _ in range(self.max_iterations):
            response_block = super().generate(
                prompt=self.memory if self.auto_update_memory else prompt,
                system=system,
                tools=tool_metadata,
                config=config,
                auto_update_memory=False,  # Agent has separate memory update
                **kwargs,
            )

            if self.auto_update_memory:
                self.memory.append(response_block.message.model_dump())

            if response_block.stop_reason == StopReason.TOOL_USE:
                yield response_block, None
                if not response_block.message:
                    raise Exception(
                        "No tool call request from the model. "
                        "Error from API bedrock when "
                        "the model is not return a valid "
                        "tool response, but still return "
                        "StopReason as TOOLUSE request."
                    )

                tool_content = (
                    response_block.message.content
                    if not response_block.message.tool_calls
                    else response_block.message.tool_calls
                )

                result = self.__process_tools(
                    cast(Union[List[ToolCallBlock], List[ToolUseBlock]], tool_content)
                )

                if isinstance(result, list):
                    if self.auto_update_memory:
                        self.memory.extend(result)
                    yield None, [r.content for r in result]
                else:
                    if self.auto_update_memory:
                        self.memory.append(result.model_dump())
                    yield None, result.content
            else:
                return response_block, None

    @lru_cache(maxsize=32)  # Cache memory updates for identical prompts
    def _get_memory_update(self, prompt_str: str) -> Dict[str, Any]:
        return MessageBlock(role="user", content=prompt_str).model_dump()

    def _manage_memory(self) -> None:
        if len(self._conversation_history) > self._memory_limit:
            self._conversation_history = self._conversation_history[
                -self._memory_limit :
            ]

    def _update_memory(
        self, prompt: Union[str, MessageBlock, Sequence[MessageBlock]]
    ) -> None:
        """Update the memory with the given prompt."""
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
