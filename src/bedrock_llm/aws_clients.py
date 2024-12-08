"""Module for managing AWS client instances."""

from typing import Any, Dict, Optional

import boto3
from botocore.config import Config


class AWSClientManager:
    """Manages both sync and async AWS client instances."""

    _sync_clients: Dict[str, Any] = {}
    _session = None

    @classmethod
    def get_sync_client(
        cls, region_name: str, profile_name: Optional[str] = None, **kwargs
    ) -> Any:
        """Get or create a cached synchronous Bedrock client."""
        cache_key = f"{region_name}_{hash(frozenset(kwargs.items()))}"
        if cache_key not in cls._sync_clients:
            config = Config(
                retries={"max_attempts": 3, "mode": "standard"},
                max_pool_connections=50,
                tcp_keepalive=True,
            )
            session = (
                boto3.Session(profile_name=profile_name)
                if profile_name
                else boto3.Session()
            )
            cls._sync_clients[cache_key] = session.client(
                "bedrock-runtime", region_name=region_name, config=config, **kwargs
            )
        return cls._sync_clients[cache_key]
