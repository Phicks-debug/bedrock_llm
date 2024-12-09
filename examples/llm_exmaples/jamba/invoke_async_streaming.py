import asyncio
from bedrock_llm import AsyncClient, ModelName


async def main():
    async with AsyncClient(
        region_name="us-east-1", model_name=ModelName.JAMBA_1_5_LARGE
    ) as client:
        async for res in client.generate_async("Write a story about a dog name Sigma."):
            print(res[0], end="", flush=True)


asyncio.run(main())
