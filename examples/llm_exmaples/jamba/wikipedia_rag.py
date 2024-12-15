import wikipedia

from bedrock_llm import AsyncClient, ModelConfig, ModelName
from typing import List, Dict, Any


# Create a LLM client
client = AsyncClient(
    region_name="us-east-1",
    model_name=ModelName.JAMBA_1_5_LARGE,
)

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.1, top_p=0.9, max_tokens=2048)


def wikipedia_retrieve(query: str) -> List[Dict[str, Any]]:
    try:
        # Search for pages
        search_results = wikipedia.search(query)
        if not search_results:
            return [{"metadata": {"title": "No Title"}, "content": "No results found."}]

        # Get the first 3 pages (or less if fewer results)
        pages_info = []
        for page_title in search_results[:3]:
            try:
                page = wikipedia.page(page_title)
                pages_info.append(
                    {"metadata": {"title": page.title}, "content": page.content}
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation by taking the first option
                page = wikipedia.page(e.options[0])
                pages_info.append(
                    {"metadata": {"title": page.title}, "content": page.content}
                )

        return pages_info

    except Exception as e:
        return [{"metadata": {"title": ""}, "content": f"Error occurred: {str(e)}"}]


async def pipeline():
    await client.init_session()

    print("RETRIEVING DOCUMENT ...")
    documents = wikipedia_retrieve("What is the capital of France?")

    print("GENERATING RESPONSE ...")
    result = client.generate_async(
        prompt="Detail of the newest large language model product release from Google.",
        documents=documents,
        config=config,
        auto_update_memory=False,
    )

    print("RESPONSE: ", end="")
    async for token, _ in result:
        if token:
            print(token, end="")

    await client.close()
    print("\nDONE")


if __name__ == "__main__":
    import asyncio

    asyncio.run(pipeline())
