"""Message schema definitions."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from ..schema.cache import CacheControl


class UserMetadata(BaseModel):
    """An object describing metadata about the request."""

    user_id: Optional[str] = Field(
        description="External identifier for the user \
            who is associated with the request",
        default=None,
    )


class Image(BaseModel):
    """An image represented as base64 data."""

    type: str = Field(description="Type of image, must be 'base_64'")
    media_type: str = Field(description="Media type of the image")
    data: str = Field(description="Base64-encoded image data")


class SystemBlock(BaseModel):
    """System prompt."""

    cache_control: Optional[CacheControl] = Field(
        description="Controls caching behavior for the system prompt", default=None
    )
    type: str = Field(description="Type of block, must be 'text'")
    text: str = Field(description="Actual system prompt text content")


class TextBlock(BaseModel):
    """Text block."""

    cache_control: Optional[CacheControl] = Field(
        description="Controls caching behavior for the text block", default=None
    )
    type: str = Field(description="Type of block, must be 'text'")
    text: str = Field(description="Actual text content")

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)


class ImageBlock(BaseModel):
    """Image block."""

    cache_control: Optional[CacheControl] = Field(
        description="Controls caching behavior for the image block", default=None
    )
    type: str = Field(description="Type of block, must be 'image'")
    source: Image = Field(description="Source of the image")


class ToolUseBlock(BaseModel):
    """Tool use block."""

    cache_control: Optional[CacheControl] = Field(
        description="Controls caching behavior for the tool use block", default=None
    )
    type: str = Field(description="Type of block, must be 'tool_use'")
    id: str = Field(description="ID of the tool use")
    name: str = Field(description="Name of the tool use")
    input: Dict = Field(description="Input to the tool use")

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)


class ToolResultBlock(BaseModel):
    """Tool result block."""

    cache_control: Optional[CacheControl] = Field(
        description="Controls caching behavior for the tool result block", default=None
    )
    type: str = Field(description="Type of block, must be 'tool_result'")
    tool_use_id: str = Field(
        description="ID of the tool use associated with this result"
    )
    is_error: bool = Field(description="Whether the tool call resulted in an error")
    content: Union[TextBlock, ImageBlock, str] = Field(
        description="Content of the tool result"
    )

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)


class ToolCallBlock(BaseModel):
    """Tool call block for the Jamba Model assistant role."""

    id: str = Field(description="ID of the tool call")
    type: Optional[str] = Field(description="Type of the function", default=None)
    function: Dict[str, Any] = Field(description="Function call to make")


class MessageBlock(BaseModel):
    """Input messages. **Only for Chat Model**"""

    role: str = Field(
        description="Role of the message sender \
        (e.g. user, assistant)"
    )
    content: Optional[
        Union[
            List[
                Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock, Dict, str]
            ],
            str,
            List[str],
        ]
    ] = Field(description="Content of the message")
    name: Optional[str] = Field(
        description="Name of the function that you return when calling", default=None
    )
    tool_calls: Optional[List[ToolCallBlock]] = Field(
        description="List of tool calls to make", default=None
    )
    tool_call_id: Optional[str] = Field(
        description="ID of the tool call after running the tool", default=None
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # If all elements are strings, keep it as is
            if all(isinstance(x, str) for x in v):
                return v
            # Otherwise, validate each element
            return [
                (
                    x
                    if isinstance(
                        x,
                        (
                            TextBlock,
                            ToolUseBlock,
                            ToolResultBlock,
                            ImageBlock,
                            Dict,
                            str,
                        ),
                    )
                    else TextBlock(type="text", text=str(x))
                )
                for x in v
            ]
        return v

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump_json(**kwargs)
