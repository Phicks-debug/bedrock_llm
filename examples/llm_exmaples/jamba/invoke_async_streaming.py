import asyncio
from bedrock_llm import AsyncClient, ModelName


async def main():
    async with AsyncClient(
        region_name="us-east-1", model_name=ModelName.JAMBA_1_5_LARGE
    ) as client:
        async for token, _ in client.generate_async(
            "Viết câu truyện về một con chó tên Sigma."
        ):
            print(token, end="", flush=True)


asyncio.run(main())
