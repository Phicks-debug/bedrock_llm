import json
import logging
import os
from typing import (Any, AsyncGenerator, Coroutine, Dict, List, Optional,
                    Tuple, Union)

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config.model import ModelConfig
from ..schema.message import MessageBlock, SystemBlock
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
        **kwargs
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
        **kwargs
    ) -> Dict[str, Any]:
        if tools:
            raise ValueError(
                """
                Titan models are not support function callings and tools.
                Please use another models
                """
            )

        if isinstance(system, SystemBlock):
            system = system.text

        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system)

        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p,
                "stopSequences": config.stop_sequences,
            },
        }

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
            return message, StopReason.END_TURN
        elif chunk["results"][0]["completionReason"] == "LENGTH":
            return message, StopReason.MAX_TOKENS
        elif chunk["results"][0]["completionReason"] == "STOP":
            return message, StopReason.STOP_SEQUENCE
        else:
            return message, StopReason.ERROR

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_response = []
        async for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["outputText"], None, None
            full_response.append(chunk["outputText"])
            if chunk["completionReason"]:
                message = MessageBlock(role="assistant", content="".join(full_response))
                if chunk["completionReason"] == "FINISH":
                    yield None, StopReason.END_TURN, message
                elif chunk["completionReason"] == "LENGTH":
                    yield None, StopReason.MAX_TOKENS, message
                elif chunk["completionReason"] == "STOP":
                    yield None, StopReason.STOP_SEQUENCE, message
                else:
                    yield None, StopReason.ERROR, message
                return


class TitanEmbedding(BaseEmbeddingsImplementation):
    def parse_embedding_response(
        self,
        response: Any
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        body = response.get("body").read()
        response_json = json.loads(body)

        if "embedding" in response_json:
            embedding = response_json["embedding"]
            metadata = {
                k: v for k,
                v in response_json.items() if k != "embedding"
            }
            return embedding, metadata
        else:
            raise ValueError("No embeddings found in response")

    async def parse_embedding_response_async(
        self,
        response: Any
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        body = response.get("body").read()
        response_json = json.loads(body)

        if "embedding" in response_json:
            embedding = response_json["embedding"]
            metadata = {
                k: v for k,
                v in response_json.items() if k != "embedding"
            }
            return embedding, metadata
        else:
            raise ValueError("No embeddings found in response")


class TitanEmbeddingsV1Implementation(TitanEmbedding):

    def prepare_embedding_request(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, List):
            raise ValueError(
                """Titan embedding model only support string as input
                Only input texts as a string that you want to embedding"""
            )

        return {
            "inputText": texts
        }

    async def prepare_embedding_request_async(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        return self.prepare_embedding_request(
            texts,
            **kwargs
        )


class TitanEmbeddingsV2Implementation(TitanEmbedding):

    def prepare_embedding_request(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        **kwargs
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
            "normalize": kwargs.pop("normalize", True)
        }

    async def prepare_embedding_request_async(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = float,
        **kwargs
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        return self.prepare_embedding_request(
            texts,
            input_type,
            **kwargs
        )


class NovaImplementation(BaseModelImplementation):
    """Implementation for Amazon's Nova model family (Micro, Lite, Pro)."""

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if tools:
            raise ValueError("Nova models do not support function calling and tools.")

        messages = []
        
        # Add system message if provided
        if system:
            if isinstance(system, SystemBlock):
                system = system.text
            messages.append({"role": "system", "content": system})
        
        # Handle prompt
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages.extend(prompt)
        else:
            raise ValueError("Prompt must be either a string or a list of message dictionaries")

        request = {
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if config.stop_sequences:
            request["stop_sequences"] = config.stop_sequences
            
        return request

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return self.prepare_request(config, prompt, system, tools, **kwargs)

    def parse_response(self, response: Any) -> Tuple[MessageBlock, StopReason]:
        if isinstance(response, bytes):
            response = json.loads(response)
            
        message = MessageBlock(
            role="assistant",
            content=response["completion"],
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        
        stop_reason_map = {
            "stop_sequence": StopReason.STOP_SEQUENCE,
            "max_tokens": StopReason.MAX_TOKENS,
            "error": StopReason.ERROR
        }
        
        return message, stop_reason_map.get(response["stop_reason"], StopReason.END_TURN)

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_response = []
        async for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            
            if "delta" in chunk:
                yield chunk["delta"], None, None
                full_response.append(chunk["delta"])
                
            if chunk.get("stop_reason"):
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response),
                    name=None,
                    tool_calls=None,
                    tool_call_id=None,
                )
                
                stop_reason_map = {
                    "stop_sequence": StopReason.STOP_SEQUENCE,
                    "max_tokens": StopReason.MAX_TOKENS,
                    "error": StopReason.ERROR
                }
                
                yield None, stop_reason_map.get(chunk["stop_reason"], StopReason.END_TURN), message
                return
