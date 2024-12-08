"""
Module for manage Streaming Parser
"""

import asyncio
import base64
import re
from typing import Dict

import orjson


class Parser:
    def __init__(self):
        # Compile the regex response parserfor efficiency
        self.json_pattern = re.compile(r'{"bytes":"(.*?)"}')

    async def parse_chunk_async(self, chunk: bytes) -> Dict:
        return await asyncio.to_thread(self.parse_chunk, chunk)

    def parse_chunk(self, chunk: bytes) -> Dict:
        # Decode the chunk with 'ignore' to skip invalid bytes
        message = chunk.decode("utf-8", errors="ignore")

        # Find all JSON objects in the message
        matches = self.json_pattern.finditer(message)

        for m in matches:
            p_out = orjson.loads(m.group(0))
            # Decode the response json contents.
            bytes = base64.b64decode(p_out["bytes"])
            p_in = orjson.loads(bytes)
            return p_in

        # Return nothing if nothing is parsed
        return {}
