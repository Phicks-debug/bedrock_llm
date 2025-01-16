from bedrock_llm import AsyncClient, ModelName, ModelConfig
from bedrock_llm.schema import MessageBlock


async def main():
    async with AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.LLAMA_3_1_70B,
    ) as client:
        async for chunk, response in client.generate_async(
            prompt=[
                MessageBlock(
                    role="user",
                    content="Kể cho tôi câu truyện về chú chó tên Sigma",
                )
            ],
            system="Your name is Chucky",
            config=ModelConfig(temperature=0.7, max_tokens=512, top_p=0.9),
            optimized=True,
        ):
            if chunk:
                print(chunk, end="", flush=True)
            if response:
                print("\n", response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
