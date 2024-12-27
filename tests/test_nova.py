"""Tests for Nova model implementation."""

import json
import pytest
from unittest.mock import MagicMock

from bedrock_llm.models.amazon import NovaImplementation
from bedrock_llm.config.model import ModelConfig
from bedrock_llm.schema.message import MessageBlock, SystemBlock
from bedrock_llm.types.enums import StopReason


def test_nova_prepare_request():
    """Test Nova prepare_request method."""
    impl = NovaImplementation()
    config = ModelConfig(
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        stop_sequences=["stop"]
    )

    # Test with string prompt
    prompt = "Hello, how are you?"
    request = impl.prepare_request(config, prompt)
    assert request == {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["stop"]
    }

    # Test with system message and prompt
    system = SystemBlock(type="text", text="You are a helpful assistant.")
    request = impl.prepare_request(config, prompt, system=system)
    assert request == {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["stop"]
    }

    # Test with message list
    messages = [
        MessageBlock(role="user", content="Hello"),
        MessageBlock(role="assistant", content="Hi!"),
        MessageBlock(role="user", content="How are you?")
    ]
    request = impl.prepare_request(config, messages)
    assert request == {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["stop"]
    }


def test_nova_parse_response():
    """Test Nova parse_response method."""
    impl = NovaImplementation()

    # Test successful completion
    response = {
        "content": [{"text": "I'm doing well, thank you!"}],
        "stop_reason": "end_turn"
    }
    message, stop_reason = impl.parse_response(json.dumps(response))
    assert message.role == "assistant"
    assert message.content == "I'm doing well, thank you!"
    assert stop_reason == StopReason.END_TURN

    # Test max tokens reached
    response = {
        "content": [{"text": "This is a long response..."}],
        "stop_reason": "max_tokens"
    }
    message, stop_reason = impl.parse_response(json.dumps(response))
    assert message.content == "This is a long response..."
    assert stop_reason == StopReason.MAX_TOKENS

    # Test stop sequence reached
    response = {
        "content": [{"text": "Stopping here"}],
        "stop_reason": "stop_sequence"
    }
    message, stop_reason = impl.parse_response(json.dumps(response))
    assert message.content == "Stopping here"
    assert stop_reason == StopReason.STOP_SEQUENCE


@pytest.mark.asyncio
async def test_nova_parse_stream_response():
    """Test Nova parse_stream_response method."""
    impl = NovaImplementation()

    # Create mock stream events
    events = [
        {"chunk": {"bytes": json.dumps({"content": [{"text": "Hello"}]})}},
        {"chunk": {"bytes": json.dumps({"content": [{"text": " world"}]})}},
        {"chunk": {"bytes": json.dumps({
            "content": [{"text": "!"}],
            "stop_reason": "end_turn"
        })}}
    ]

    # Create async generator mock
    async def mock_stream():
        for event in events:
            yield event

    # Test streaming response
    chunks = []
    async for chunk, stop_reason, message in impl.parse_stream_response(mock_stream()):
        if chunk:
            chunks.append(chunk)
        if stop_reason:
            assert stop_reason == StopReason.END_TURN
            assert message.content == "Hello world!"

    assert chunks == ["Hello", " world", "!"]


def test_nova_tools_not_supported():
    """Test that tools are not supported in Nova models."""
    impl = NovaImplementation()
    config = ModelConfig()
    prompt = "Hello"
    tools = [{"type": "function", "function": {"name": "test"}}]

    with pytest.raises(ValueError, match="Nova models do not support function calling and tools"):
        impl.prepare_request(config, prompt, tools=tools)
