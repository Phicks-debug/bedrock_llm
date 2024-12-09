import asyncio
import wikipedia

from bedrock_llm import Agent, ModelName
from bedrock_llm.schema import ToolMetadata, InputSchema, PropertyAttr
from bedrock_llm.schema.message import MessageBlock


wikipedia_tool = ToolMetadata(
    name="search_wikipedia",
    description="Search Wikipedia and return page content",
    input_schema=InputSchema(
        type="object",
        properties={
            "query": PropertyAttr(
                type="string",
                description="The query to search Wikipedia",
            )
        },
        required=["query"],
    ),
)


@Agent.tool(wikipedia_tool)
async def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return page content"""
    page = wikipedia.page(query)
    return page.content


async def main():
    async with Agent(region_name="us-east-1", model_name=ModelName.NOVA_PRO) as client:
        async for res in client.generate_and_action_async(
            prompt=MessageBlock(
                role="user", content="What is the newst Large Language Model from AWS."
            ),
            system="You must search on wikipedia for information",
            tools=["search_wikipedia"],
        ):
            print(res[0], end="", flush=True)


asyncio.run(main())
