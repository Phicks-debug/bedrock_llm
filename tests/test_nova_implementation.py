import pytest
import json
from bedrock_llm.models.amazon import NovaImplementation
from bedrock_llm.config.model import ModelConfig
from bedrock_llm.schema.message import MessageBlock

def test_nova_prepare_request():
    nova = NovaImplementation()
    config = ModelConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=[]
    )
    
    # Test with simple prompt
    prompt = "Hello, how are you?"
    request = nova.prepare_request(config, prompt)
    
    assert "messages" in request
    assert isinstance(request["messages"], list)
    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"] == prompt
    
    # Test with system message
    system = "You are a helpful assistant"
    request = nova.prepare_request(config, prompt, system=system)
    
    assert len(request["messages"]) == 2
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][0]["content"] == system
    
    # Test with message history
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    request = nova.prepare_request(config, messages)
    
    assert len(request["messages"]) == 3
    assert request["messages"] == messages

def test_nova_parse_response():
    nova = NovaImplementation()
    
    # Test normal completion
    response = {
        "completion": "I'm doing well, thank you!",
        "stop_reason": "stop_sequence"
    }
    
    message, stop_reason = nova.parse_response(response)
    assert message.role == "assistant"
    assert message.content == "I'm doing well, thank you!"
    assert stop_reason.name == "STOP_SEQUENCE"

def test_nova_parse_stream_response():
    nova = NovaImplementation()
    
    # Mock streaming response
    async def mock_stream():
        yield {"chunk": {"bytes": json.dumps({
            "delta": "Hello",
            "stop_reason": None
        })}}
        yield {"chunk": {"bytes": json.dumps({
            "delta": " there",
            "stop_reason": None
        })}}
        yield {"chunk": {"bytes": json.dumps({
            "delta": "!",
            "stop_reason": "stop_sequence"
        })}}
    
    async def test_streaming():
        full_response = []
        async for text, stop_reason, message in nova.parse_stream_response(mock_stream()):
            if text:
                full_response.append(text)
            if stop_reason:
                assert stop_reason.name == "STOP_SEQUENCE"
                assert message.content == "Hello there!"
                
    import asyncio
    asyncio.run(test_streaming())
