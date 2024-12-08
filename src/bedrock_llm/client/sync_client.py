"""Sync client implementation."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from ..aws_clients import AWSClientManager
from ..client.base import BaseClient
from ..config.base import RetryConfig
from ..config.model import ModelConfig
from ..schema.message import MessageBlock
from ..schema.tools import ToolMetadata
from ..types.enums import ModelName, StopReason


class Client(BaseClient):
    """Sync client for Bedrock LLM implementations."""

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        memory: Optional[List[MessageBlock]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_iterations: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize sync client."""
        super().__init__(
            region_name,
            model_name,
            memory,
            retry_config,
            max_iterations,
            **kwargs,
        )
        profile_name = kwargs.pop("profile_name", None)
        self._sync_client = AWSClientManager.get_sync_client(
            region_name,
            profile_name,
            **kwargs,
        )

    def generate(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
        config: Optional[ModelConfig] = None,
        auto_update_memory: bool = True,
        **kwargs: Any,
    ) -> Tuple[MessageBlock, StopReason]:
        """Generate a response from the model synchronously."""
        config_internal = config or ModelConfig()
        invoke_message = self._process_prompt(prompt, auto_update_memory)

        request_body = self.model_implementation.prepare_request(
            config=config_internal,
            prompt=cast(
                Union[str, List[Dict[Any, Any]]],
                invoke_message,
            ),
            system=system,
            tools=tools,
            **kwargs,
        )

        response = self._handle_retry_logic_sync(
            self._invoke_model_sync,
            client=self._sync_client,
            request_body=request_body,
        )
        response_msg, stop_reason = self.model_implementation.parse_response(
            response["body"].read()
        )

        if self.memory is not None and auto_update_memory and response_msg is not None:
            self.memory.append(response_msg.model_dump())

        return response_msg, stop_reason

    def _invoke_model_sync(self, client: Any, request_body: Dict[str, Any]) -> Any:
        """Invoke model with sync client."""
        return client.invoke_model(
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(request_body),
            trace="ENABLED",
        )
