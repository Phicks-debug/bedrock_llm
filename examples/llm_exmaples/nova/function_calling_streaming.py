import asyncio

from bedrock_llm import AsyncClient, ModelName
from bedrock_llm.schema import ToolMetadata, InputSchema, PropertyAttr

weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the weather for a given location usiing AccuWeather API",
    input_schema=InputSchema(
        properties={
            "location": PropertyAttr(
                type="string",
                description="The location for which to get the weather",
            ),
            "unit": PropertyAttr(
                type="string",
                enum=["celsius", "fahrenheit"],
                description="The unit of temperature",
            ),
        },
        required=["location"],
    ),
)


async def main():
    client = AsyncClient(region_name="us-east-1", model_name=ModelName.NOVA_PRO)

    await client.init_session()
    result = client.generate_async(
        prompt="What is the weather in Hanoi and Paris?",
        system="You have a weather tool to check weather",
        tools=[weather_tool],
    )
    async for token, stop, full_response in result:
        if token:
            print(token, end="", flush=True)
        if stop:
            print(f"\n{stop.name}")
        if full_response:
            print(f"\n{full_response}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
