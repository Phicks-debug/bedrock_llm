"""Anthropic model implementation."""

import asyncio
import json
from typing import (Any, AsyncGenerator, Coroutine, Dict, List, Optional,
                    Tuple, Union, cast)

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import (ImageBlock, MessageBlock, SystemBlock, TextBlock,
                              ToolResultBlock, ToolUseBlock)
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason


class ClaudeImplementation(BaseModelImplementation):
    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict[Any, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict[Any, Any]]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(prompt, str):
            prompt = [MessageBlock(role="user", content=prompt).model_dump()]
        elif isinstance(prompt, list):
            prompt = [
                msg.model_dump() if isinstance(msg, MessageBlock) else msg
                for msg in prompt
            ]

        request_body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.max_tokens,
            "messages": prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop_sequences": config.stop_sequences,
        }

        if system is not None:
            request_body["system"] = (
                system.text.strip()
                if isinstance(system, SystemBlock)
                else system.strip()
            )

        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice

        return request_body

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict[Any, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict[Any, Any]]]] = None,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, **kwargs
        )

    def parse_response(self, response: Any) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response)
        message = MessageBlock(
            role=chunk["role"],
            content=chunk["content"],
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        if chunk.get("stop_reason") == "end_turn":
            return message, StopReason.END_TURN
        elif chunk.get("stop_reason") == "stop_sequence":
            return message, StopReason.STOP_SEQUENCE
        elif chunk.get("stop_reason") == "max_token":
            return message, StopReason.MAX_TOKENS
        elif chunk.get("stop_reason") == "tool_use":
            return message, StopReason.TOOL_USE
        return message, StopReason.ERROR

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_response = ""
        tool_input = ""
        message = MessageBlock(
            role="assistant",
            content=cast(
                List[Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock]], []
            ),
        )

        async for chunk in stream:
            if chunk["type"] == "content_block_delta":
                if chunk["delta"]["type"] == "text_delta":
                    text_chunk = chunk["delta"]["text"]
                    yield text_chunk, None, None
                    full_response += text_chunk
                elif chunk["delta"]["type"] == "input_json_delta":
                    text_chunk = chunk["delta"]["partial_json"]
                    tool_input += text_chunk

            elif chunk["type"] == "content_block_start":
                id = chunk["content_block"].get("id")
                name = chunk["content_block"].get("name")

            elif chunk["type"] == "content_block_stop":
                if full_response:
                    if isinstance(message.content, list):
                        message.content.append(
                            TextBlock(type="text", text=full_response)
                        )
                    full_response = ""
                else:
                    try:
                        input_data = json.loads(tool_input)
                    except json.JSONDecodeError:
                        input_data = {}
                    tool = ToolUseBlock(
                        type="tool_use", id=id, name=name, input=input_data
                    )
                    if isinstance(message.content, list):
                        message.content.append(tool)
                    tool_input = ""

            elif chunk["type"] == "message_delta":
                stop_reason = chunk["delta"]["stop_reason"]
                if stop_reason:
                    if stop_reason == "end_turn":
                        yield None, StopReason.END_TURN, message
                    elif stop_reason == "stop_sequence":
                        yield None, StopReason.STOP_SEQUENCE, message
                    elif stop_reason == "max_tokens":
                        yield None, StopReason.MAX_TOKENS, message
                    elif stop_reason == "tool_use":
                        yield None, StopReason.TOOL_USE, message
                    else:
                        yield None, StopReason.ERROR, message
                    return
