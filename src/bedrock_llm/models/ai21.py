import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import MessageBlock, SystemBlock
from ..schema.response import ResponseBlock, TraceBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason


class JambaImplementation(BaseModelImplementation):

    def _parse_tool_metadata(
        self, tool: Union[ToolMetadata, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse a ToolMetadata object or a dictionary
            into the format required by the Mistral model.
        """

        if isinstance(tool, dict):
            # Handle all dictionary inputs consistently
            if "type" in tool and tool["type"] == "function":
                function_data = tool.get("function", {})
            else:
                function_data = tool

            return {
                "type": "function",
                "function": {
                    "name": function_data.get("name", "unnamed_function"),
                    "description": function_data.get(
                        "description", "No description provided"
                    ),
                    "parameters": function_data.get(
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                },
            }

        if isinstance(tool, ToolMetadata):
            jamba_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

            if tool.input_schema:
                # Convert properties Collection to a dictionary
                properties = {}
                for name, attr in tool.input_schema.properties.items():
                    # Convert Pydantic model to dict
                    attr_dict = attr.model_dump()
                    properties[name] = {
                        "type": attr_dict["type"],
                        "description": attr_dict["description"],
                    }

                # Add properties to parameters
                jamba_tool["function"]["parameters"]["properties"] = properties

                # Convert required Collection to a list if it exists
                if tool.input_schema.required is not None:
                    jamba_tool["function"]["parameters"]["required"] = list(
                        tool.input_schema.required
                    )

            return jamba_tool

        raise ValueError(
            f"Unsupported tool type: {type(tool)}. Expected Dict or ToolMetadata."
        )

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare the request body for the AI21 API.

        Args:
            prompt (str | List[Dict]): The prompt to send to the AI21 API.
            config (ModelConfig): The configuration for the AI21 API.
            system (Optional[str]): The system prompt to send to the AI21 API.
            documents (Optional[str]): The context documents to send to the AI21 API.
            tools (Optional[List[Dict] | Dict]): The tools to send to the AI21 API.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The request body for the AI21 API.

        Raises:
            ValueError: If the prompt is not a string or a list of dictionaries.
            ValueError: If the instruction is not a string.
            ValueError: If tools are provided (not supported).
            ValueError: If documents are provided (not supported).

        See more: https://docs.ai21.com/docs/prompt-engineering
        """
        messages = []

        if isinstance(prompt, str):
            messages.append(MessageBlock(role="user", content=prompt).model_dump())
        else:
            messages.extend(prompt)

        if system is not None:
            if isinstance(system, SystemBlock):
                system = system.text
            system = MessageBlock(role="system", content=system).model_dump()
            messages.insert(0, system)

        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "stop": config.stop_sequences,
            "n": config.number_of_responses,
        }

        # Conditionally add document if provided
        if kwargs.get("documents"):
            if not isinstance(kwargs.get("documents"), List):
                raise ValueError(
                    """Documents must be a list of dict.
Please read this for more information of Document use:
https://docs.ai21.com/reference/jamba-15-api-ref"""
                )
            request_body["documents"] = kwargs.get("documents")

        # Conditionally add response format if provided
        if kwargs.get("response_format"):
            if not isinstance(kwargs.get("documents"), List):
                raise ValueError(
                    """Documents must be a list of dict.
Please read this for more information of Document use:
https://docs.ai21.com/reference/jamba-15-api-ref"""
                )
            request_body["response_format"] = kwargs.get("response_format")

        # Conditionally add tools if provided
        if tools:
            parsed_tools = []
            for tool in tools:
                if isinstance(tool, (dict, ToolMetadata)):
                    parsed_tools.append(self._parse_tool_metadata(tool))
                else:
                    raise ValueError(
                        f"""Unsupported tool type in list: {type(tool)}.
                        Expected Dict or ToolMetadata."""
                    )

            request_body["tools"] = parsed_tools

        return request_body

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare the request body for the AI21 API.

        Args:
            prompt (str | List[Dict]): The prompt to send to the AI21 API.
            config (ModelConfig): The configuration for the AI21 API.
            system (Optional[str]): The system prompt to send to the AI21 API.
            documents (Optional[str]): The context documents to send to the AI21 API.
            tools (Optional[List[Dict] | Dict]): The tools to send to the AI21 API.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The request body for the AI21 API.

        Raises:
            ValueError: If the prompt is not a string or a list of dictionaries.
            ValueError: If the instruction is not a string.

        See more: https://docs.ai21.com/docs/prompt-engineering
        """
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, **kwargs
        )

    @staticmethod
    def _extract_chunk_data(chunk: dict) -> tuple[Optional[str], Optional[str]]:
        """Extract text content and stop reason from a chunk."""
        if not chunk.get("choices"):
            return None, None

        choice = chunk["choices"][0]
        return (choice["delta"].get("content"), choice.get("finish_reason"))

    def parse_response(self, response: Any) -> ResponseBlock:
        block = json.loads(response)
        chunk = block["choices"][0]
        message = MessageBlock(
            role="assistant",
            content=(
                chunk["message"]["content"].strip()
                if chunk["message"]["content"]
                else None
            ),
            tool_calls=chunk["message"].get("tool_calls", None),
            name=None,
            tool_call_id=None,
        )
        if chunk.get("finish_reason") == "stop":
            stop_reason = StopReason.END_TURN
        elif chunk.get("finish_reason") == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.ERROR

        trace = TraceBlock(
            input_tokens=block["usage"]["prompt_tokens"],
            output_tokens=block["usage"]["completion_tokens"],
            total_tokens=block["usage"]["total_tokens"],
            metadata=block["meta"],
        )

        return ResponseBlock(
            id=block["id"],
            message=message,
            stop_reason=stop_reason,
            trace=trace,
        )

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[ResponseBlock]], None]:
        """
        Parse the response from the Bedrock API, handling both text content
        and tool call requests.

        Args:
            stream: The response stream from the Bedrock API.

        Yields:
            Tuple containing either:
            - (str, None): Regular text chunks
            - (MessageBlock, str): Final message(optional tool calls), stop reason
        """
        full_answer: List[str] = []

        async for chunk in stream:
            try:
                text_chunk, stop_reason = self._extract_chunk_data(chunk)

                if stop_reason:
                    trace = TraceBlock(
                        input_tokens=chunk["usage"]["prompt_tokens"],
                        output_tokens=chunk["usage"]["completion_tokens"],
                        total_tokens=chunk["usage"]["total_tokens"],
                        metadata=chunk["meta"],
                    )
                    message = MessageBlock(
                        role="assistant", content="".join(full_answer).strip()
                    )
                    if stop_reason == "stop":
                        stop = StopReason.END_TURN
                    elif stop_reason == "length":
                        stop = StopReason.MAX_TOKENS
                    yield None, ResponseBlock(
                        message=message,
                        stop_reason=stop,
                        trace=trace,
                    )
                    return

                if not text_chunk:
                    continue

                if not stop_reason:
                    full_answer.append(text_chunk)
                    yield text_chunk, None

            except Exception as e:
                print(f"Unexpected error processing chunk: {str(e)}")
                continue
