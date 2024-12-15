"""Anthropic model implementation."""

import asyncio
import json
from typing import (Any, AsyncGenerator, Coroutine, Dict, List, Optional,
                    Tuple, Union, cast)

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import (ImageBlock, MessageBlock, SystemBlock, TextBlock,
                              ToolResultBlock, ToolUseBlock)
from ..schema.response import ResponseBlock, TraceBlock
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

    def parse_response(self, response: Any) -> ResponseBlock:
        chunk = json.loads(response)
        message = MessageBlock(
            role=chunk["role"],
            content=chunk["content"],
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        if chunk.get("stop_reason") == "end_turn":
            stop_reason = StopReason.END_TURN
        elif chunk.get("stop_reason") == "stop_sequence":
            stop_reason = StopReason.STOP_SEQUENCE
        elif chunk.get("stop_reason") == "max_token":
            stop_reason = StopReason.MAX_TOKENS
        elif chunk.get("stop_reason") == "tool_use":
            stop_reason = StopReason.TOOL_USE
        else:
            stop_reason = StopReason.ERROR

        trace = TraceBlock(
            input_tokens=chunk["usage"]["input_tokens"],
            output_tokens=chunk["usage"]["output_tokens"],
            total_tokens=chunk["usage"]["input_tokens"]
            + chunk["usage"]["output_tokens"],
        )

        return ResponseBlock(
            id=chunk["id"],
            message=message,
            stop_reason=stop_reason,
            trace=trace,
        )

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[ResponseBlock]], None]:
        full_response = []
        tool_input = []
        stop = StopReason.END_TURN

        async for chunk in stream:
            if not chunk:
                continue

            if chunk["type"] == "message_start":
                id_mess = chunk["message"]["id"]
                message = MessageBlock(
                    role=chunk["message"]["role"],
                    content=cast(
                        List[
                            Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock]
                        ],
                        [],
                    ),
                    tool_calls=None,
                    tool_call_id=None,
                )

            elif chunk["type"] == "content_block_delta":
                if chunk["delta"]["type"] == "text_delta":
                    text_chunk = chunk["delta"]["text"]
                    full_response.append(text_chunk)
                    yield text_chunk, None
                elif chunk["delta"]["type"] == "input_json_delta":
                    text_chunk = chunk["delta"]["partial_json"]
                    tool_input.append(text_chunk)

            elif chunk["type"] == "content_block_start":
                id = chunk["content_block"].get("id")
                name = chunk["content_block"].get("name")

            elif chunk["type"] == "content_block_stop":
                if full_response:
                    message.content.append(
                        TextBlock(type="text", text="".join(full_response))
                    )
                    full_response.clear()
                if tool_input:
                    try:
                        input_data = json.loads("".join(tool_input))
                    except json.JSONDecodeError:
                        input_data = {}
                    tool = ToolUseBlock(
                        type="tool_use", id=id, name=name, input=input_data
                    )
                    if isinstance(message.content, list):
                        message.content.append(tool)
                    tool_input.clear()

            elif chunk["type"] == "message_delta":
                stop_reason = chunk["delta"]["stop_reason"]
                if stop_reason:
                    if stop_reason == "end_turn":
                        stop = StopReason.END_TURN
                    elif stop_reason == "stop_sequence":
                        stop = StopReason.STOP_SEQUENCE
                    elif stop_reason == "max_tokens":
                        stop = StopReason.MAX_TOKENS
                    elif stop_reason == "tool_use":
                        stop = StopReason.TOOL_USE
                    else:
                        stop = StopReason.ERROR

            elif chunk["type"] == "message_stop":
                metrics = chunk["amazon-bedrock-invocationMetrics"]
                trace = TraceBlock(
                    input_tokens=metrics["inputTokenCount"],
                    output_tokens=metrics["outputTokenCount"],
                    total_tokens=metrics["inputTokenCount"]
                    + metrics["outputTokenCount"],
                    metadata={
                        "invocation_latency": metrics["invocationLatency"],
                        "first_byte_latency": metrics["firstByteLatency"],
                    },
                )
                yield None, ResponseBlock(
                    id=id_mess,
                    message=message,
                    stop_reason=stop,
                    trace=trace,
                )
                return
