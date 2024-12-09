import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import (MessageBlock, SystemBlock, TextBlock,
                              ToolCallBlock)
from ..schema.response import ResponseBlock, TraceBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason, ToolChoiceEnum


class MistralChatImplementation(BaseModelImplementation):
    """
    Read more:
    https://docs.aws.amazon.com/bedrock/latest/userguide
    """

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
            mistral_tool = {
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
                mistral_tool["function"]["parameters"]["properties"] = properties

                # Convert required Collection to a list if it exists
                if tool.input_schema.required is not None:
                    mistral_tool["function"]["parameters"]["required"] = list(
                        tool.input_schema.required
                    )

            return mistral_tool

        raise ValueError(
            f"Unsupported tool type: {type(tool)}. Expected Dict or ToolMetadata."
        )

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if tools and not isinstance(tools, (list, dict, ToolMetadata)):
            raise ValueError(
                "Tools must be a list, dictionary, or ToolMetadata object."
            )

        messages = []
        if isinstance(prompt, str):
            messages.append(MessageBlock(role="user", content=prompt).model_dump())
        elif isinstance(prompt, list):
            messages.extend(prompt)

        if system is not None:
            if isinstance(system, SystemBlock):
                system = system.text
            system = MessageBlock(role="system", content=system).model_dump()
            messages.insert(0, system)

        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        # Conditionally add tools and tool_choice if they are not None
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
        prompt: Union[str, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, tool_choice, **kwargs
        )

    def parse_response(self, response: Any) -> ResponseBlock:
        block = json.loads(response)
        chunk = block["choices"][0]
        message = MessageBlock(
            role=chunk["message"]["role"],
            content=chunk["message"]["content"],
            tool_calls=(
                chunk["message"]["tool_calls"]
                if "tool_calls" in chunk["message"]
                else None
            ),
            tool_call_id=(
                chunk["message"]["tool_call_id"]
                if "tool_call_id" in chunk["message"]
                else None
            ),
        )
        if chunk["finish_reason"] == "stop":
            stop_reason = StopReason.END_TURN
        elif chunk["finish_reason"] == "tool_calls":
            stop_reason = StopReason.TOOL_USE
        elif chunk["finish_reason"] == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.ERROR

        trace = TraceBlock(
            input_tokens=block["usage"]["prompt_tokens"],
            output_tokens=block["usage"]["completion_tokens"],
            total_tokens=block["usage"]["total_tokens"],
        )

        return ResponseBlock(
            id=block["id"], message=message, stop_reason=stop_reason, trace=trace
        )

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_response: List[str] = []
        async for chunk in stream:
            chunk = chunk["choices"][0]
            if chunk["stop_reason"]:
                content = "".join(full_response) if full_response else ""
                message = MessageBlock(
                    role="assistant",
                    content=(
                        [TextBlock(type="text", text=content)] if content else None
                    ),
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "tool_calls":
                    if "tool_calls" in chunk["message"]:
                        tool_calls = [
                            ToolCallBlock(
                                id=tool_call["id"],
                                type=tool_call["type"],
                                function=tool_call["function"],
                            )
                            for tool_call in chunk["message"]["tool_calls"]
                        ]
                        message.tool_calls = tool_calls
                    yield None, StopReason.TOOL_USE, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                if "content" in chunk["message"] and chunk["message"]["content"]:
                    yield chunk["message"]["content"], None, None
                    full_response.append(chunk["message"]["content"])
                elif "tool_calls" in chunk["message"]:
                    # Handle streaming tool calls
                    tool_calls = [
                        ToolCallBlock(
                            id=tool_call["id"],
                            type=tool_call["type"],
                            function=tool_call["function"],
                        )
                        for tool_call in chunk["message"]["tool_calls"]
                    ]
                    message = MessageBlock(
                        role="assistant",
                        content=[TextBlock(type="text", text="")],
                        tool_calls=tool_calls,
                    )
                    yield None, None, message


class MistralInstructImplementation(BaseModelImplementation):
    """
    Read more: https://docs.mistral.ai/guides/prompting_capabilities/
    """

    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")

    def load_template(
        self, prompt: Union[MessageBlock, List[Dict]], system: Optional[str]
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml", "j2"]),
        )
        template = env.get_template("mistral7_template.j2")
        return template.render({"SYSTEM": system, "REQUEST": prompt})

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if tools:
            raise ValueError(
                "Mistral 7B Instruct does not support tools. Please use another model."
            )

        system_content = system.text if isinstance(system, SystemBlock) else system

        formatted_prompt = (
            self.load_template(prompt, system_content)
            if not isinstance(prompt, str)
            else prompt
        )

        return {
            "prompt": formatted_prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        }

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, tool_choice, **kwargs
        )

    def parse_response(self, response: Any) -> ResponseBlock:
        chunk = json.loads(response)
        chunk = chunk["outputs"][0]
        message = MessageBlock(role="assistant", content=chunk["text"])
        if chunk["stop_reason"] == "stop":
            stop_reason = StopReason.END_TURN
        elif chunk["stop_reason"] == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.ERROR
        return ResponseBlock(
            message=message,
            stop_reason=stop_reason,
            trace=TraceBlock(),
        )

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_response: List[str] = []
        async for event in stream:
            chunk = event["outputs"][0]
            if chunk["stop_reason"]:
                message = MessageBlock(role="assistant", content="".join(full_response))
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                yield chunk["text"], None, None
                full_response.append(chunk["text"])
