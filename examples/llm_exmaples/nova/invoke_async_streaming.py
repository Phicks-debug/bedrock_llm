import asyncio
from bedrock_llm import AsyncClient, ModelName


async def main():
    async with AsyncClient(
        region_name="us-east-1", model_name=ModelName.NOVA_MICRO
    ) as client:
        async for res in client.generate_async(
            "Viết câu truyện về một con chó tên Sigma."
        ):
            print(res[0], end="", flush=True)
            if res[1]:
                print(res[1])


asyncio.run(main())
