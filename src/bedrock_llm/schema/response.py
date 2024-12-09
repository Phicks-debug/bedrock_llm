"""Response Schema Definition"""

import secrets
import string
from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..types.enums import StopReason
from .message import MessageBlock


def generate_message_id(prefix="msg_bdrk_", length=24):
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{random_part}"


class TraceBlock(BaseModel):
    """Trace Schema Definition"""

    input_tokens: Optional[int] = Field(
        description="The number of input tokens", default=None
    )
    output_tokens: Optional[int] = Field(
        description="The number of output tokens", default=None
    )
    total_tokens: Optional[int] = Field(
        description="The total number of tokens", default=None
    )
    cache_read_input_token: Optional[int] = Field(
        description="The number of input tokens read from cache", default=None
    )
    cache_write_input_token: Optional[int] = Field(
        description="The number of input tokens written to cache", default=None
    )
    metadata: Optional[Dict] = Field(
        description="The metadata of the response", default=None
    )


class ResponseBlock(BaseModel):
    """Response Block for synchronous"""

    id: str = Field(description="The id of the response", default=generate_message_id())
    message: MessageBlock = Field(description="The message from the model")
    stop_reason: StopReason = Field(description="The reason why the model stopped")
    trace: TraceBlock = Field(
        description="The trace of the request from AWS Bedrock Client"
    )


class StreamResponseChunk(BaseModel):
    """Stream Response Chunk for asynchronous"""

    token: str = Field(description="The streaming token ouput from the model")
    stop_reason: StopReason = Field(description="The reason why the model stopped")
    message: MessageBlock = Field(
        description="The final full message response from the model"
    )
    trace: TraceBlock = Field(
        description="The trace of the request from AWS Bedrock Client"
    )
