from enum import Enum


class ModelName(str, Enum):
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_5_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_3_5_SONNET = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    LLAMA_3_1_8B = "us.meta.llama3-1-8b-instruct-v1:0"
    LLAMA_3_1_70B = "us.meta.llama3-1-70b-instruct-v1:0"
    LLAMA_3_1_405B = "meta.llama3-1-405b-instruct-v1:0"
    LLAMA_3_2_1B = "us.meta.llama3-2-1b-instruct-v1:0"
    LLAMA_3_2_3B = "us.meta.llama3-2-3b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"
    LLAMA_3_2_90B = "us.meta.llama3-2-90b-instruct-v1:0"
    LLAMA_3_3_70B = "us.meta.llama3-3-70b-instruct-v1:0"
    TITAN_LITE = "amazon.titan-text-lite-v1"
    TITAN_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_PREMIER = "amazon.titan-text-premier-v1:0"
    JAMBA_1_5_MINI = "ai21.jamba-1-5-mini-v1:0"
    JAMBA_1_5_LARGE = "ai21.jamba-1-5-large-v1:0"
    MISTRAL_LARGE_2 = "mistral.mistral-large-2407-v1:0"
    MISTRAL_7B = "mistral.mistral-7b-instruct-v0:2"
    TITAN_EMBED_V1 = "amazon.titan-embed-text-v1"
    TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
    TITAN_EMBED_IMAGE = "amazon.titan-embed-image-v1"
    COHERE_ENG = "cohere.embed-english-v3"
    COHERE_MULTI = "cohere.embed-multilingual-v3"
    COHERE_RERANK_3_5 = "cohere.rerank-v3-5:0"
    TITAN_IMAGE_GEN = "amazon.titan-embed-image-v1"
    STABLE_DIFF_3_LARGE = "stability.sd3-large-v1:0"
    STABLE_IMAGE_CORE = "stability.stable-image-core-v1:0"
    STABLE_IMAGE_ULTRA = "stability.stable-image-ultra-v1:0"
    STABLE_DIFF_XL = "stability.stable-diffusion-xl-v1"
    RERANK_1 = "amazon.rerank-v1:0"
    NOVA_MICRO = "amazon.nova-micro-v1:0"
    NOVA_LITE = "amazon.nova-lite-v1:0"
    NOVA_PRO = "amazon.nova-pro-v1:0"
    NOVA_CANVAS = "amazon.nova-canvas-v1:0"
    NOVA_REEL = "amazon.nova-reel-v1:0"


class ToolChoiceEnum(Enum):
    AUTO = "auto"
    ANY = "any"
    NONE = "none"


class StopReason(Enum):
    END_TURN = 1
    MAX_TOKENS = 2
    STOP_SEQUENCE = 3
    TOOL_USE = 4
    ERROR = 5


class ToolState(Enum):
    CLAUDE = 1  # Claude way of tool calling
    MISTRAL_JAMBA_LLAMA = 2  # Mistral AI, Jamba, Llama way of tool calling
