import asyncio
from bedrock_llm import EmbedClient, ModelName


def sync_main():
    sync_client = EmbedClient(
        region_name="us-east-1",
        model_name=ModelName.COHERE_ENG,
    )
    text = "Hello, this is sync function"
    response = sync_client.embed(text, "search_document")
    print(response)


async def main():
    async_client = EmbedClient(
        region_name="us-east-1",
        model_name=ModelName.TITAN_EMBED_V2,
    )

    text = "Hello, this is a sample text for embedding async"
    response = await async_client.embed_async(text, "search_document",  dimensions=256)
    print(response)
    await async_client.close()


if __name__ == "__main__":
    print("Sync")
    sync_main()
    print("Async")
    asyncio.run(main())
