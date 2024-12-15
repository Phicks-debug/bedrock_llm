# Add for print console with color
from termcolor import cprint

from bedrock_llm import AsyncAgent, ModelConfig, ModelName
from bedrock_llm.schema import InputSchema, PropertyAttr, ToolMetadata, MessageBlock
from bedrock_llm.types.enums import StopReason

system = """You are a helpful assistant.
You have access to realtime information.
You can use tools to get the real time data weather of a city."""

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.1, top_p=0.9, max_tokens=512)

# Create tool definition
get_weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the real time weather of a city",
    input_schema=InputSchema(
        type="object",
        properties={
            "location": PropertyAttr(
                type="string", description="The city to get the weather of"
            )
        },
        required=["location"],
    ),
)

# Create user prompt
prompt = MessageBlock(
    role="user", content="What is the weather in New York and Toronto?"
)


@AsyncAgent.tool(get_weather_tool)
async def get_weather(location: str):
    # Mock function to get weather
    return f"tools_result: {location} is 20*C"


async def main():
    # Create agent and init session
    async with AsyncAgent(
        region_name="us-west-2", model_name=ModelName.CLAUDE_3_5_HAIKU
    ) as agent:
        # Invoke agent
        async for (
            token,
            response_block,
            tool_result,
        ) in agent.generate_and_action_async(
            prompt=prompt,
            system=system,
            tools=[get_weather],
            config=config,
        ):
            # Print out the token
            if token:
                cprint(token, "green", end="", flush=True)

            # Print out the tool request from the model
            if response_block:
                if response_block.stop_reason == StopReason.TOOL_USE:
                    cprint(
                        f"\n{response_block.message.content}",
                        "cyan",
                        end="",
                        flush=True,
                    )
                    cprint(f"\n{response_block.stop_reason}", "red")
                else:
                    cprint(f"\n{response_block.stop_reason}", "red")

            # Print out the result from executed the tool
            if tool_result:
                cprint(f"\n{tool_result}", "yellow")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
