# flake8: noqa: E203
"""Meta model implementation."""

import asyncio
import json
import logging
import os
import uuid
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union)

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models.base import (BaseModelImplementation, MessageBlock, ModelConfig,
                           SystemBlock)
from ..schema.response import ResponseBlock, TraceBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason


class LlamaImplementation(BaseModelImplementation):
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")

    def load_template(
        self,
        prompt: Sequence[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[Sequence[ToolMetadata]] = None,
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml", "j2"]),
        )
        template = env.get_template("llama32_template.j2")
        rendered = template.render(
            {"SYSTEM": system, "REQUEST": prompt, "TOOLS": tools}
        )
        return rendered

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Sequence[ToolMetadata]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(system, SystemBlock):
            system = system.text

        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, tools)

        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Sequence[ToolMetadata]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, **kwargs
        )

    def parse_response(self, response: Any) -> ResponseBlock:
        chunk = json.loads(response)
        response_text = chunk["generation"].strip()

        if response_text.startswith("[") and response_text.endswith("]"):
            message = MessageBlock(role="tool", content=response_text)
            stop_reason = StopReason.TOOL_USE
        else:
            message = MessageBlock(role="assistant", content=response_text)
            if chunk["stop_reason"] == "stop":
                stop_reason = StopReason.END_TURN
            elif chunk["stop_reason"] == "length":
                stop_reason = StopReason.MAX_TOKENS
            else:
                stop_reason = StopReason.ERROR

        trace = TraceBlock(
            input_tokens=chunk["prompt_token_count"],
            output_tokens=chunk["generation_token_count"],
            total_tokens=chunk["prompt_token_count"] + chunk["generation_token_count"],
        )

        return ResponseBlock(
            message=message,
            stop_reason=stop_reason,
            trace=trace,
        )

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse Llama's string format tool calls into structured format.

        Args:
            response (str): String containing tool calls like
                "[get_weather(location='New York')]"

        Returns:
            List[Dict[str, Any]]: List of parsed tool calls in standard format
        """
        content = response[1:-1].strip()
        if not content:
            return []

        # Split by commas not inside parentheses
        depth = 0
        current = []
        calls = []

        for char in content:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                calls.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        if current:
            calls.append("".join(current).strip())

        tool_calls = []
        for call in calls:
            # Extract function name and arguments
            func_name = call[: call.index("(")]
            args_str = call[call.index("(") + 1 : call.rindex(")")]

            # Parse arguments
            args = {}
            if args_str:
                for arg in args_str.split(","):
                    key, value = arg.split("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    args[key] = value

            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": func_name.strip(),
                        "arguments": json.dumps(args),
                    },
                }
            )

        return tool_calls

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[ResponseBlock]], None]:
        full_answer: List[str] = []
        is_stop = False
        message = None

        async for chunk in stream:
            if not is_stop:
                full_answer.append(chunk["generation"])
                yield chunk["generation"], None

            if not is_stop:  # Make sure the yield did not execute dead logic
                if chunk.get("stop_reason"):
                    response = "".join(full_answer).strip()

                    # Handle empty response
                    if not response:
                        message = MessageBlock(role="assistant", content="")
                        stop = StopReason.END_TURN
                        is_stop = True
                        continue

                    # Check if response is a tool call
                    if response.startswith("[") and response.endswith("]"):
                        try:
                            tool_calls = self._parse_tool_calls(response)
                            if tool_calls:
                                message = MessageBlock(
                                    role="assistant",
                                    content="<|python_tag|>" + response,
                                    tool_calls=tool_calls,
                                )
                                stop = StopReason.TOOL_USE
                                is_stop = True
                            else:
                                message = MessageBlock(
                                    role="assistant", content=response
                                )
                                stop = StopReason.ERROR
                                is_stop = True
                        except Exception as e:
                            logging.error(f"Failed to parse tool calls: {e}")
                            message = MessageBlock(role="assistant", content=response)
                            stop = StopReason.ERROR
                            is_stop = True
                    else:
                        message = MessageBlock(role="assistant", content=response)
                        if chunk["stop_reason"] == "stop":
                            stop = StopReason.END_TURN
                        elif chunk["stop_reason"] == "length":
                            stop = StopReason.MAX_TOKENS
                        else:
                            stop = StopReason.ERROR

            if chunk.get("amazon-bedrock-invocationMetrics") and message:
                metrics = chunk["amazon-bedrock-invocationMetrics"]
                yield None, ResponseBlock(
                    message=message,
                    stop_reason=stop,
                    trace=TraceBlock(
                        input_tokens=metrics["inputTokenCount"],
                        output_tokens=metrics["outputTokenCount"],
                        total_tokens=metrics["inputTokenCount"]
                        + metrics["outputTokenCount"],
                        metadata={
                            "invocation_latency": metrics["invocationLatency"],
                            "first_byte_latency": metrics["firstByteLatency"],
                        },
                    ),
                )
                return
