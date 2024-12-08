# Add for print console with color
from termcolor import cprint

from bedrock_llm import Agent, ModelConfig, ModelName, RetryConfig
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


@Agent.tool(get_weather_tool)
async def get_weather(location: str):
    # Mock function to get weather
    return f"tools_result: {location} is 20*C"


async def main():
    # Create agent and init session
    async with Agent(
        region_name="us-east-1",
        model_name=ModelName.CLAUDE_3_5_HAIKU,
        retry_config=RetryConfig(max_attempts=3),
    ) as agent:
        # Invoke agent
        async for (
            token,
            stop_reason,
            response,
            tool_result,
        ) in agent.generate_and_action_async(
            prompt=prompt,
            system=system,
            tools=["get_weather"],
            config=config,
        ):
            # Print out the token
            if token:
                cprint(token, "green", end="", flush=True)

            # Print out the tool request from the model
            if stop_reason == StopReason.TOOL_USE:
                cprint(f"\n{response.content}", "cyan", end="", flush=True)
                cprint(f"\n{stop_reason}", "red")

            # Print out the result from executed the tool
            if tool_result:
                cprint(f"\n{tool_result}", "yellow")

            # Print out the stop reason
            elif stop_reason:
                cprint(f"\n{stop_reason}", "red")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
