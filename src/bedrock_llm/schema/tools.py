"""Tool Schema Definitions"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..schema.cache import CacheControl


class PropertyAttr(BaseModel):
    """Property attributes for tool parameters."""

    type: Literal["string", "integer", "float", "boolean", "array"] = Field(
        description="Data type of the parameter"
    )
    enum: Optional[List[str]] = Field(
        description="Possible enum values",
        default=None,
    )
    description: str = Field(description="Description of the parameter")

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)

    def __hash__(self) -> int:
        return hash(
            (self.type, tuple(self.enum) if self.enum else None, self.description)
        )


class InputSchema(BaseModel):
    """JSON schema for tool input."""

    type: Literal["object"] = Field(
        description="Type of input schema",
        default="object",
    )
    properties: Dict[str, PropertyAttr] = Field(
        description="Tool input properties schema",
        default_factory=dict,
    )
    required: Optional[List[str]] = Field(
        description="Required fields",
        default=None,
    )

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)

    def __hash__(self) -> int:
        return hash(
            (
                self.type,
                tuple(sorted(self.properties.items())),
                tuple(sorted(self.required)) if self.required else None,
            )
        )


class ToolMetadata(BaseModel):
    """Tool metadata including description and parameters."""

    type: Optional[str] = Field(
        description="Type of tool, read more Anthropic's documentation",
        default=None,
    )
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    input_schema: InputSchema = Field(
        description="Tool input schema",
        default_factory=InputSchema,
    )
    cache_control: Optional[CacheControl] = Field(
        description="Cache control for the tool",
        default=None,
    )

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(**kwargs)

    def __hash__(self) -> int:
        return hash(
            (
                self.type,
                self.name,
                self.description,
                self.input_schema,
                self.cache_control,
            )
        )
