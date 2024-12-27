"""Model implementations for Bedrock LLM."""

from typing import Dict, Type, Union

from ..types.enums import ModelName
from .ai21 import JambaImplementation
from .amazon import (TitanEmbeddingsV1Implementation,
                     TitanEmbeddingsV2Implementation, TitanImplementation)
from .anthropic import ClaudeImplementation
from .base import BaseModelImplementation
from .cohere import CohereEnglishEmbedding, CohereMultilingualEmbedding
from .embeddings import BaseEmbeddingsImplementation
from .meta import LlamaImplementation
from .mistral import MistralChatImplementation, MistralInstructImplementation

MODEL_IMPLEMENTATIONS: Dict[
    ModelName, Union[Type[BaseModelImplementation],
                     Type[BaseEmbeddingsImplementation]]
] = {
    ModelName.CLAUDE_3_HAIKU: ClaudeImplementation,
    ModelName.CLAUDE_3_5_HAIKU: ClaudeImplementation,
    ModelName.CLAUDE_3_5_SONNET: ClaudeImplementation,
    ModelName.CLAUDE_3_5_OPUS: ClaudeImplementation,
    ModelName.LLAMA_3_2_1B: LlamaImplementation,
    ModelName.LLAMA_3_2_3B: LlamaImplementation,
    ModelName.LLAMA_3_2_11B: LlamaImplementation,
    ModelName.LLAMA_3_2_90B: LlamaImplementation,
    ModelName.TITAN_LITE: TitanImplementation,
    ModelName.TITAN_EXPRESS: TitanImplementation,
    ModelName.TITAN_PREMIER: TitanImplementation,
    ModelName.NOVA_MICRO: TitanImplementation,
    ModelName.NOVA_LITE: TitanImplementation,
    ModelName.NOVA_PRO: TitanImplementation,
    ModelName.JAMBA_1_5_LARGE: JambaImplementation,
    ModelName.JAMBA_1_5_MINI: JambaImplementation,
    ModelName.MISTRAL_7B: MistralInstructImplementation,
    ModelName.MISTRAL_LARGE_2: MistralChatImplementation,
    ModelName.TITAN_EMBED_V1: TitanEmbeddingsV1Implementation,
    ModelName.TITAN_EMBED_V2: TitanEmbeddingsV2Implementation,
    ModelName.COHERE_ENG: CohereEnglishEmbedding,
    ModelName.COHERE_MULTI: CohereMultilingualEmbedding,
}

__all__ = [
    "BaseModelImplementation",
    "BaseEmbeddingsImplementation",
    "TitanEmbeddingsImplementation",
    "TitanImplementation",
    "ClaudeImplementation",
    "JambaImplementation",
    "LlamaImplementation",
    "MistralChatImplementation",
    "MistralInstructImplementation",
    "CohereEnglishEmbedding",
    "CohereMultilingualEmbedding",
    "MODEL_IMPLEMENTATIONS"
]
