# flake8: noqa: E501
import asyncio
import json
import logging
import os
from typing import (Any, AsyncGenerator, Coroutine, Dict, List, Optional,
                    Tuple, Union)

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config.model import ModelConfig
from ..schema.message import (ImageBlock, MessageBlock, SystemBlock, TextBlock,
                              ToolUseBlock)
from ..schema.response import ResponseBlock, TraceBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason
from .base import BaseModelImplementation
from .embeddings import (BaseEmbeddingsImplementation, EmbeddingInputType,
                         EmbeddingVector, Metadata)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TitanImplementation(BaseModelImplementation):
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")

    def load_template(
        self, prompt: Union[MessageBlock, List[Dict]], system: Optional[str]
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml", "j2"]),
        )
        template = env.get_template("amazon_template.j2")
        return template.render({"SYSTEM": system, "REQUEST": prompt}).strip() + " "

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if tools:
            raise ValueError(
                """
                Amazon Titan models do not support function calling and tools.
                Please use another model.
                """
            )

        if isinstance(system, SystemBlock):
            system = system.text

        formatted_prompt = (
            self.load_template(prompt, system)
            if not isinstance(prompt, str)
            else prompt
        )

        return {
            "inputText": formatted_prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p,
                "stopSequences": config.stop_sequences,
            },
        }

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, **kwargs
        )

    def parse_response(self, response: Any) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response)
        message = MessageBlock(
            role="assistant",
            content=chunk["results"][0]["outputText"],
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        if chunk["results"][0]["completionReason"] == "FINISH":
            stop_reason = StopReason.END_TURN
        elif chunk["results"][0]["completionReason"] == "LENGTH":
            stop_reason = StopReason.MAX_TOKENS
        elif chunk["results"][0]["completionReason"] == "STOP":
            stop_reason = StopReason.STOP_SEQUENCE
        else:
            stop_reason = StopReason.ERROR
        trace = TraceBlock(
            input_tokens=chunk["inputTextTokenCount"],
            output_tokens=chunk["results"][0]["tokenCount"],
            total_tokens=chunk["inputTextTokenCount"]
            + chunk["results"][0]["tokenCount"],
        )
        return ResponseBlock(
            message=message,
            stop_reason=stop_reason,
            trace=trace,
        )

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[ResponseBlock]], None]:
        full_response = []
        message = None
        is_stop = False

        async for chunk in stream:
            if not is_stop:
                full_response.append(chunk["outputText"])

                if chunk["completionReason"]:
                    message = MessageBlock(
                        role="assistant", content="".join(full_response)
                    )
                    if chunk["completionReason"] == "FINISH":
                        stop = StopReason.END_TURN
                    elif chunk["completionReason"] == "LENGTH":
                        stop = StopReason.MAX_TOKENS
                    elif chunk["completionReason"] == "STOP":
                        stop = StopReason.STOP_SEQUENCE
                    else:
                        stop = StopReason.ERROR
                yield chunk["outputText"], None

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


class TitanEmbedding(BaseEmbeddingsImplementation):
    def parse_embedding_response(
        self, response: Any
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        body = response.get("body").read()
        response_json = json.loads(body)

        if "embedding" in response_json:
            embedding = response_json["embedding"]
            metadata = {k: v for k, v in response_json.items() if k != "embedding"}
            return embedding, metadata
        else:
            raise ValueError("No embeddings found in response")

    async def parse_embedding_response_async(
        self, response: Any
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        body = response.get("body").read()
        response_json = json.loads(body)

        if "embedding" in response_json:
            embedding = response_json["embedding"]
            metadata = {k: v for k, v in response_json.items() if k != "embedding"}
            return embedding, metadata
        else:
            raise ValueError("No embeddings found in response")


class TitanEmbeddingsV1Implementation(TitanEmbedding):
    def prepare_embedding_request(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, List):
            raise ValueError(
                """Titan embedding model only support string as input
                Only input texts as a string that you want to embedding"""
            )

        return {"inputText": texts}

    async def prepare_embedding_request_async(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        return await asyncio.to_thread(self.prepare_embedding_request, texts, **kwargs)


class TitanEmbeddingsV2Implementation(TitanEmbedding):
    def prepare_embedding_request(
        self, texts: Union[str, List[str]], input_type: EmbeddingInputType, **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, List):
            raise ValueError(
                """Titan embedding model only support string as input
                Only input texts as a string that you want to embedding"""
            )

        if input_type != "search_document":
            logging.warning(
                """This model only support 1 type of input.
                'search_document'"""
            )

        return {
            "inputText": texts,
            "dimensions": kwargs.pop("dimensions", 1024),
            "normalize": kwargs.pop("normalize", True),
        }

    async def prepare_embedding_request_async(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = float,
        **kwargs,
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        return await asyncio.to_thread(
            self.prepare_embedding_request, texts, input_type, **kwargs
        )


class NovaImplementation(BaseModelImplementation):
    def convert_to_nova_format(
        self, message: Union[MessageBlock, str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert MessageBlock to Nova format."""
        nova_content = []

        if isinstance(message, str):
            nova_content.append({"text": message})
            return {"role": "user", "content": nova_content}

        if isinstance(message, dict):
            # Handle dictionary input
            content = message.get("content", "")
            role = message.get("role", "user")

            if isinstance(content, str):
                nova_content.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            nova_content.append({"text": item["text"]})
                        elif "image" in item:
                            nova_content.append({"image": item["image"]})
                    elif isinstance(item, str):
                        nova_content.append({"text": item})

            return {"role": role, "content": nova_content}

        elif isinstance(message, MessageBlock):
            if isinstance(message.content, str):
                nova_content.append({"text": message.content})
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, str):
                        nova_content.append({"text": item})
                    elif isinstance(item, TextBlock):
                        nova_content.append({"text": item.text})
                    elif isinstance(item, ImageBlock):
                        nova_content.append(
                            {
                                "image": {
                                    "format": item.source.media_type.split("/")[-1],
                                    "source": {"bytes": item.source.data},
                                }
                            }
                        )
            return {"role": message.role, "content": nova_content}

        raise ValueError(f"Unsupported message type: {type(message)}")

    def convert_to_nova_tool(self, tools: List[ToolMetadata]) -> List[Dict[str, Dict]]:
        """Convert tools to Nova tools format."""
        t: List[Dict[str, Dict]] = []
        for tool in tools:
            if isinstance(tool, ToolMetadata):
                # Convert ToolMetadata to Nova's format
                nova_tool = {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {
                            "json": {
                                "type": tool.input_schema.type,
                                "properties": {
                                    name: {
                                        "type": prop.type,
                                        "description": prop.description,
                                        **({"enum": prop.enum} if prop.enum else {}),
                                    }
                                    for name, prop in tool.input_schema.properties.items()
                                },
                                "required": tool.input_schema.required or [],
                            }
                        },
                    }
                }
                t.append(nova_tool)
            else:
                # If it's already a dict, assume it's in the correct format
                t.append({"toolSpec": tool})
        return t

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict[Any, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs,
    ):
        """Prepare request for Nova model."""

        # Convert messages to Nova format
        if isinstance(prompt, List):
            nova_messages = [self.convert_to_nova_format(msg) for msg in prompt]
        else:
            nova_messages = [self.convert_to_nova_format(prompt)]

        body_request = {
            "messages": nova_messages,
            "inferenceConfig": {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "stopSequences": config.stop_sequences,
            },
        }

        # Prepare and convert tools to Nova format
        if tools:
            t = self.convert_to_nova_tool(tools)
            body_request["toolConfig"] = {
                "tools": t,
                "toolChoice": {
                    "auto": {}
                },  # Amazon Nova models ONLY support tool choice of "auto"
            }

        # Prepare system message
        system_content = None
        if system:
            if isinstance(system, str):
                system_content = [{"text": system}]
            elif isinstance(system, SystemBlock):
                system_content = [{"text": system.text}]
            body_request["system"] = system_content

        return body_request

    async def prepare_request_async(
        self, config, prompt, system=None, tools=None, **kwargs
    ):
        return await asyncio.to_thread(
            self.prepare_request, config, prompt, system, tools, **kwargs
        )

    def parse_response(self, response: Any):
        isToolUse = False
        message = MessageBlock(role="assistant", content=[])
        chunk = json.loads(response)
        content = chunk["output"]["message"]["content"]
        for i in content:
            if i.get("text"):
                message.content.append(TextBlock(type="text", text=i["text"]))
            if i.get("toolUse"):
                isToolUse = True
                tool = i["toolUse"]
                message.content.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool["toolUseId"],
                        name=tool["name"],
                        input=tool["input"],
                    )
                )
        if chunk["stopReason"] == "end_turn":
            stop_reason = StopReason.END_TURN
            if isToolUse:
                stop_reason = StopReason.TOOL_USE
        if chunk["stopReason"] == "error":
            stop_reason = StopReason.ERROR
        elif stop_reason == "stop_sequence":
            stop_reason = StopReason.STOP_SEQUENCE
        elif stop_reason == "max_tokens":
            stop_reason = StopReason.MAX_TOKENS
        usage = TraceBlock(
            input_tokens=chunk["usage"]["inputTokens"],
            output_tokens=chunk["usage"]["outputTokens"],
            total_tokens=chunk["usage"]["totalTokens"],
            cache_read_input_token=chunk["usage"]["cacheReadInputTokenCount"],
            cache_write_input_token=chunk["usage"]["cacheWriteInputTokenCount"],
        )
        return ResponseBlock(
            message=message,
            stop_reason=stop_reason,
            trace=usage,
        )

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[ResponseBlock]], None]:
        """Parse the streaming response from Nova model."""
        full_response = []
        message = MessageBlock(
            role="assistant",
            content=[],
        )
        current_tool_use = None
        tool_uses = []

        async for chunk in stream:
            print(chunk)
            # Handle message start
            if "messageStart" in chunk:
                continue

            # Handle content block start (for tool use)
            if "contentBlockStart" in chunk:
                if "toolUse" in chunk["contentBlockStart"]["start"]:
                    current_tool_use = {
                        "type": "tool_use",
                        "id": chunk["contentBlockStart"]["start"]["toolUse"][
                            "toolUseId"
                        ],
                        "name": chunk["contentBlockStart"]["start"]["toolUse"]["name"],
                        "input": {},
                    }

            # Handle content block delta (text and tool use)
            if "contentBlockDelta" in chunk:
                if "text" in chunk["contentBlockDelta"]["delta"]:
                    text_chunk = chunk["contentBlockDelta"]["delta"]["text"]
                    full_response.append(text_chunk)
                    yield text_chunk, None,
                elif "toolUse" in chunk["contentBlockDelta"]["delta"]:
                    if current_tool_use:
                        input_str = chunk["contentBlockDelta"]["delta"]["toolUse"][
                            "input"
                        ]
                        current_tool_use["input"] = json.loads(input_str)

            # Handle content block stop
            elif "contentBlockStop" in chunk:
                if current_tool_use:
                    tool_uses.append(ToolUseBlock(**current_tool_use))
                    current_tool_use = None
                continue

            elif "messageStop" in chunk:
                if tool_uses:
                    stop = StopReason.TOOL_USE
                elif chunk["messageStop"]["stopReason"] == "end_turn":
                    stop = StopReason.END_TURN
                elif chunk["messageStop"]["stopReason"] == "stop_sequence":
                    stop = StopReason.STOP_SEQUENCE
                elif chunk["messageStop"]["stopReason"] == "max_tokens":
                    stop = StopReason.MAX_TOKENS
                else:
                    stop = StopReason.ERROR

            # Handle end of response with metadata
            elif "metadata" in chunk:
                if full_response:
                    usage = TraceBlock(
                        input_tokens=chunk["metadata"]["usage"]["inputTokens"],
                        output_tokens=chunk["metadata"]["usage"]["outputTokens"],
                        total_tokens=chunk["metadata"]["usage"]["inputTokens"]
                        + chunk["metadata"]["usage"]["outputTokens"],
                        cache_read_input_token=chunk["metadata"]["usage"].get(
                            "cacheReadInputTokenCount"
                        ),
                        cache_write_input_token=chunk["metadata"]["usage"].get(
                            "cacheWriteInputTokenCount"
                        ),
                        metadata=chunk["amazon-bedrock-invocationMetrics"],
                    )

            # Handle empty final chunk
            elif chunk == {}:
                continue

        # If we get here without returning, yield final message
        if len(full_response) > 0:
            if isinstance(message.content, list):
                message.content.append(
                    TextBlock(type="text", text="".join(full_response))
                )
                if tool_uses:
                    message.content.extend(tool_uses)

            yield None, ResponseBlock(
                message=message,
                stop_reason=stop,
                trace=usage,
            )
