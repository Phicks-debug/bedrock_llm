# flake8: noqa: E501
"""Async client implementation."""

import asyncio
import json
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union, cast)

import aiohttp
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from ..config.base import RetryConfig
from ..config.model import ModelConfig
from ..schema.message import MessageBlock
from ..schema.response import ResponseBlock
from ..schema.tools import ToolMetadata
from ..types.enums import ModelName, StopReason
from ..types.exceptions import AsyncClientSessionError
from .base import BaseClient
from .parser import Parser


class AsyncClient(BaseClient):
    """Async client for Bedrock LLM implementations."""

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        memory: Optional[List[MessageBlock]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_iterations: Optional[int] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize async client."""
        super().__init__(
            region_name, model_name, memory, retry_config, max_iterations, **kwargs
        )
        self.service = "bedrock"
        self.session = None

        # Initialize boto3 session
        self.boto3_session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )

        self.credentials = self.boto3_session.get_credentials()
        self.res_parser = Parser()

    async def __aenter__(self):
        if not self.session:
            await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init_session(self, limit: int = 1000, ttl_dns_cache: int = 300):
        connector = aiohttp.TCPConnector(
            limit=limit,
            ttl_dns_cache=ttl_dns_cache,
            use_dns_cache=True,
            enable_cleanup_closed=True,
            verify_ssl=True,
        )
        self.session = aiohttp.ClientSession(connector=connector)

    async def close(self):
        if self.session:
            await self.session.close()

    async def _sign_request_async(self, url, method, body):
        return await asyncio.to_thread(self._sign_request, url, method, body)

    def _sign_request(self, url, method, body):
        credentials = self.credentials

        request = AWSRequest(
            method=method, url=url, data=json.dumps(body) if body else ""
        )

        request.headers.add_header("Content-Type", "application/json")
        request.headers.add_header("X-Amzn-Bedrock-Accept", "application/json")
        request.headers.add_header("X-Amzn-Bedrock-Trace", "ENABLED")
        request.headers.add_header(
            "Host", f"bedrock-runtime.{self.region_name}.amazonaws.com"
        )

        SigV4Auth(credentials, self.service, self.region_name).add_auth(request)

        return dict(request.headers)

    async def generate_async(
        self,
        prompt: Union[str, MessageBlock, Sequence[MessageBlock]],
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
        config: Optional[ModelConfig] = None,
        auto_update_memory: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[ResponseBlock]], None]:
        """Generate a response from the model asynchronously with streaming."""
        config_internal = config or ModelConfig()
        invoke_messages = self._process_prompt(prompt, auto_update_memory)

        async def _generate_stream():
            """Invoke model through aiohttp API and handle streaming event"""
            request_body = await self.model_implementation.prepare_request_async(
                config=config_internal,
                prompt=cast(
                    Union[str, List[Dict[Any, Any]]],
                    invoke_messages,
                ),
                system=system,
                tools=tools,
                **kwargs,
            )
            url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{self.model_name}/invoke-with-response-stream"

            headers = await self._sign_request_async(url, "POST", request_body)

            try:
                if not self.session:
                    raise AsyncClientSessionError("Session is not initialized.")

                async with self.session.post(
                    url, json=request_body, headers=headers
                ) as response:
                    if response.status == 200:
                        async for chunk, _ in response.content.iter_chunks():
                            yield await self.res_parser.parse_chunk_async(chunk)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Error {response.status}: {error_text}")
            except AsyncClientSessionError as e:
                raise AsyncClientSessionError
            except asyncio.TimeoutError:
                raise Exception("Request timed out.")
            except aiohttp.ClientError as e:
                raise Exception(f"Client error: {str(e)}")

        try:
            async for (
                token,
                res_block,
            ) in self.model_implementation.parse_stream_response(_generate_stream()):
                await asyncio.sleep(0)
                if (
                    self.memory is not None
                    and auto_update_memory
                    and res_block is not None
                ):
                    if isinstance(res_block, ResponseBlock):
                        self.memory.append(res_block.message.model_dump())
                yield token, res_block
        except Exception:
            async for (
                token,
                res_block,
            ) in self._handle_retry_logic_stream(
                self.model_implementation.parse_stream_response, _generate_stream()
            ):
                await asyncio.sleep(0)
                if (
                    self.memory is not None
                    and auto_update_memory
                    and res_block is not None
                ):
                    if isinstance(res_block, ResponseBlock):
                        self.memory.append(res_block.message.model_dump())
                yield token, res_block
