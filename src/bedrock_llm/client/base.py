"""Base client implementation."""

import asyncio
import json
import logging
import time
from abc import ABC
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from botocore.exceptions import ClientError, ReadTimeoutError

from ..aws_clients import AWSClientManager
from ..config.base import RetryConfig
from ..models import MODEL_IMPLEMENTATIONS, BaseModelImplementation
from ..schema.message import MessageBlock
from ..types.enums import ModelName

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMemoryException(TypeError):
    pass


class BaseClient(ABC):
    """Base client for Bedrock LLM implementations.

    This class provides common functionality used by both sync and async clients.
    """

    _model_implementations: Dict[ModelName, BaseModelImplementation] = {}
    _aws_client_manager = None

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        memory: Optional[List[MessageBlock]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_iterations: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize base client.

        Args:
            region_name: AWS region name
            model_name: Name of the model to use
            memory: Optional list of message blocks for conversation history
            retry_config: Optional retry configuration
            max_iterations: Optional maximum number of iterations
            **kwargs: Additional keyword arguments passed to AWS client
        """
        self.region_name = region_name
        self.model_name = model_name
        self.retry_config = retry_config or RetryConfig()
        self.model_implementation = self._get_or_create_model_implementation(
            model_name
        )
        self.memory = memory
        self.max_iterations = max_iterations
        self._async_client = None

    async def __aenter__(self):
        """Async context manager enter."""
        await self._get_async_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def open(self):
        """Initialize the async client if not already initialized."""
        if self._async_client is None:
            self._async_client = await self._get_async_client()

    async def _get_async_client(self):
        """Get or create async AWS client."""
        if self._async_client is None:
            self._async_client = await AWSClientManager.get_async_client(
                self.region_name
            )
        return self._async_client

    def _handle_retry_logic_sync(self, operation, *args, **kwargs):
        """Handle retry logic for sync operations."""
        for attempt in range(self.retry_config.max_retries):
            try:
                result = operation(*args, **kwargs)
                return result
            except (ReadTimeoutError, ClientError) as e:
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.retry_delay * (
                        2**attempt if self.retry_config.exponential_backoff else 1
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise
        raise Exception("Max retries reached. Unable to invoke model.")

    async def _handle_retry_logic_stream(self, operation, *args, **kwargs):
        """Handle retry logic for streaming async operations."""
        for attempt in range(self.retry_config.max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    async for result in await operation(*args, **kwargs):
                        yield result
                else:
                    async for result in operation(*args, **kwargs):
                        yield result
                return
            except (ReadTimeoutError, ClientError) as e:
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.retry_delay * (
                        2**attempt if self.retry_config.exponential_backoff else 1
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise
        raise Exception("Max retries reached. Unable to invoke model.")

    @lru_cache(maxsize=32)
    def _get_or_create_model_implementation(
        self,
        model_name: ModelName
    ) -> BaseModelImplementation:
        """Get or create a model implementation instance."""
        if model_name not in self._model_implementations:
            self._model_implementations[model_name] = (
                MODEL_IMPLEMENTATIONS[model_name]()
            )
        return self._model_implementations[model_name]

    def _process_prompt(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        auto_update_memory: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """Process the input prompt and update memory if needed."""
        if isinstance(prompt, str):
            if auto_update_memory and self.memory:
                raise ModelMemoryException(
                    """Prompt must be MessageBlock or list when
                    auto update memory is enabled"""
                )
            return prompt

        if isinstance(prompt, (MessageBlock, list)):
            if isinstance(prompt, MessageBlock):
                invoke_message = [prompt.model_dump()]
            else:
                invoke_message = [
                    m.model_dump() if isinstance(m, MessageBlock) else m
                    for m in prompt
                ]

            if self.memory is not None and auto_update_memory:
                self.memory.extend(invoke_message)

                return self.memory
            else:
                return invoke_message

        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _invoke_model_sync(
        self,
        client: Any,
        request_body: Dict[str, Any]
    ) -> Any:
        """Invoke model with sync client."""
        return client.invoke_model(
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(request_body),
        )

    async def _invoke_model_async(
        self,
        request_body: Dict[str, Any]
    ):
        client = await self._get_async_client()
        return await client.invoke_model_with_response_stream(
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(request_body),
            performanceConfigLatency = "optimized"
        )

    async def close(self):
        """Close the async client session."""
        if self._async_client:
            await self._async_client.__aexit__(None, None, None)
            self._async_client = None
        await AWSClientManager.close_async_clients()
