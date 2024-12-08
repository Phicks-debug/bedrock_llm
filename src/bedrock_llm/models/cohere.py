"""Cohere embeddings model implementations."""

import json
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union

from ..models.embeddings import EmbeddingInputType, EmbeddingVector, Metadata
from .embeddings import BaseEmbeddingsImplementation


class CohereEmbedding(BaseEmbeddingsImplementation):
    """Base class for Cohere embedding models."""

    def parse_embedding_response(
        self, response: Any
    ) -> Tuple[List[EmbeddingVector], Optional[Metadata]]:
        """Parse the embedding response from Cohere.

        Args:
            response: Raw response from the model.

        Returns:
            A tuple containing the list of embedding vectors and optional metadata.
        """
        body = response.get("body").read()
        response_json = json.loads(body)
        embeddings = response_json.get("embeddings", [])

        embedding_vectors = []
        for embedding in embeddings:
            embedding_vectors.append({"embedding_vector": embedding})

        metadata = {k: v for k, v in response_json.items() if k != "embeddings"}
        return embedding_vectors, metadata if metadata else None

    async def parse_embedding_response_async(
        self, response: Any
    ) -> Tuple[List[EmbeddingVector], Optional[Metadata]]:
        """Async version of parse_embedding_response."""
        return self.parse_embedding_response(response)


class CohereMultilingualEmbedding(CohereEmbedding):
    """Implementation for Cohere's multilingual embedding model."""

    def prepare_embedding_request(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]

        return {
            "texts": texts,
            "input_type": input_type,
            # "truncate": None,
            # "embedding_type": embedding_type,
        }

    async def prepare_embedding_request_async(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        """Async version of prepare_embedding_request."""
        return self.prepare_embedding_request(
            texts=texts, input_type=input_type, embedding_type=embedding_type, **kwargs
        )


class CohereEnglishEmbedding(CohereEmbedding):
    """Implementation for Cohere's English embedding model V3."""

    def prepare_embedding_request(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]

        return {
            "texts": texts,
            "input_type": input_type,
            # "truncate": None,
            # "embedding_type": embedding_type,
        }

    async def prepare_embedding_request_async(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        """Async version of prepare_embedding_request."""
        return self.prepare_embedding_request(
            texts=texts, input_type=input_type, embedding_type=embedding_type, **kwargs
        )
